cd Backend

python -m venv venv

venv\Scripts\Activate.ps1
pip install fastapi mangum boto3 pydantic

pip freeze > requirement.txt
rm -rf venv

pip install -t dependencies -r requirements.txt
Compress-Archive -Path dependencies\* -DestinationPath deployment_package.zip
Compress-Archive -Path api.py -Update -DestinationPath deployment_package.zip
