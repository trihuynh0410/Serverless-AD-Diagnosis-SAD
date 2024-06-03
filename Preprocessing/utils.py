import os
import tempfile

import boto3
import time
from nipype.interfaces import fsl
from nipype import Node, Workflow
import shutil


def download_file_from_s3(bucket, key, file_path, s3_client):
    print('Downloading', bucket, key, file_path)
    with open(file_path, 'wb') as file:
        s3_client.download_fileobj(bucket, key, file)
    print('Downloaded', bucket, key)

def clean_file(file_path):
    os.remove(file_path)
    print('Temporary file deleted')

def empty_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f'Folder {folder_path} has been emptied.')
    else:
        print(f'Folder {folder_path} does not exist.')

def preprocess(file_path, ref_path, base_dir):

    # reorient = Node(fsl.Reorient2Std(in_file=file_path), output_type='NIFTI_GZ', name='reorient')
    robustfov = Node(fsl.RobustFOV(in_file=file_path), output_type='NIFTI_GZ', name='robustfov')
    fast = Node(fsl.FAST(output_biascorrected = True, img_type = 1, no_pve = True), output_type='NIFTI_GZ', name='fast')
    flirt = Node(fsl.FLIRT(reference = ref_path), output_type='NIFTI_GZ', name='flirt')
    bet = Node(fsl.BET(in_file=file_path), output_type='NIFTI_GZ', name='bet')
    wf = Workflow(name="flow", base_dir=base_dir)
    wf.connect([
        # (reorient, robustfov,[('out_file','in_file')]),
        (robustfov, fast,[('out_roi','in_files')]),
        (fast, flirt,[('restored_image','in_file')]),
        (flirt, bet,[('out_file','in_file')])
    ])
    wf.run()

def naming(key, base_dir):
    file_name = os.path.basename(key)

    base_name, _ = os.path.splitext(file_name)

    file_path = base_dir + "/flow/bet/" + base_name + "_ROI_restore_flirt_brain.nii.gz"
    s3_key = "fsl/" + base_name + ".nii.gz"
    return file_path, s3_key




