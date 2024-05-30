from tqdm import tqdm

import dicom2nifti
import os
import warnings
warnings.filterwarnings("ignore")

dc_class = ["CN", "MCI", "Mild"]
nii_class = ["CN_nii", "MCI_nii", "Mild_nii"]
new_class = ["CN_new", "MCI_new", "Mild_new"]

root_dir = r""

for dc_stage, nii_stage, new_stage in zip(dc_class,nii_class, new_class):
    nii_class_dir = os.path.join(root_dir, nii_stage)
    nii_dirs = os.listdir(nii_class_dir)

    dc_class_dir = os.path.join(root_dir, dc_stage)
    dc_dirs = os.listdir(dc_class_dir)

    new_class_dir = os.path.join(root_dir, new_stage)
    new_dirs = os.listdir(dc_class_dir)
    convert = list(set(dc_dirs) - set(nii_dirs))

    for dir in tqdm(convert, desc=f"Processing {dc_stage}"):
        dc_dir = os.path.join(dc_class_dir, dir)
        nii_dir = os.path.join(new_class_dir, dir)
        # print(nii_dir)
        os.makedirs(nii_dir)

        dicom2nifti.convert_directory(dc_dir, nii_dir, compression=False, reorient=True)
        file = os.listdir(nii_dir)
        file_path = os.path.join(nii_dir, file[0])
        name = dc_stage + "_" + dir + ".nii"
        new_file_path = os.path.join(nii_dir, name)
        os.rename(file_path, new_file_path)
        # break
    # break