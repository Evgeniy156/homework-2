import boto3
import os
from botocore.client import Config

s3 = boto3.client('s3',
    endpoint_url='http://85.239.47.84:9000',
    aws_access_key_id='kalininevgeny',
    aws_secret_access_key='VeryLongRandomSecret123',
    config=Config(signature_version='s3v4')
)

bucket_name = 'mlops-data'
object_name = 'lesson4_data/data.zip'
file_name = 'data.zip'

print(f"Attempting to download {object_name} from {bucket_name}...")
try:
    s3.download_file(bucket_name, object_name, file_name)
    print("Download successful")
except Exception as e:
    print(f"Error: {e}")
