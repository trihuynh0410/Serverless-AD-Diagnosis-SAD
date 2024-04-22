import os
import tempfile

import boto3
import time
from utils import *
from nipype.interfaces import fsl
from nipype import Node, Workflow
import shutil

s3_client = boto3.client('s3')

base_dir = "/tmp"
ref_file = '/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz'
os.makedirs(base_dir, exist_ok=True)

def handler(event, context):
    try:
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print('Converting', bucket, ':', key)

        filename = os.path.basename(key)
        temp_file_path = os.path.join(base_dir, filename)
        download_file_from_s3(bucket, key, temp_file_path, s3_client)
        
        print("tmp file path:", temp_file_path)
        preprocess(temp_file_path, ref_file, base_dir)

        file_path, s3_key = naming(key, base_dir)
        
        with open(file_path, 'rb') as data:
            response = s3_client.put_object(
                Body=data,
                Bucket=bucket,
                Key=s3_key,
            )
            
        print(response)
        empty_folder(base_dir)

    except Exception as e:
        print('Error:', str(e))
