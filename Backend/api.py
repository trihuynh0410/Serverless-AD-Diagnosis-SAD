import boto3, time, os

from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse

from mangum import Mangum
from boto3.dynamodb.conditions import Key
from pydantic import BaseModel, Field
from typing import Optional
from boto3.dynamodb.types import TypeDeserializer
from dotenv import load_dotenv
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
DYNAMODB_URL = os.getenv("DYNAMODB_URL")
S3_BUCKET_NAME='thesis-ad'
s3_client = boto3.client('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

dynamodb_resource = boto3.resource('dynamodb', endpoint_url=DYNAMODB_URL)
dynamodb_client = boto3.client('dynamodb', endpoint_url=DYNAMODB_URL)

patient_info = dynamodb_resource.Table('patient_info')
fsl = dynamodb_resource.Table('fsl')

deserializer = TypeDeserializer()

app = FastAPI()

class Patient(BaseModel):
    id: int = Field(default_factory=lambda: int(time.time()))
    name: str
    dob: str
    gender: bool

class Fsl(BaseModel):
    id: int  
    day_upload: int = Field(default_factory=lambda: int(time.time()))
    name: str
    predict: Optional[int] = -1
    note: Optional[str] = None

temp_dir = '/tmp'
os.makedirs(temp_dir, exist_ok=True)
print(f"Temporary directory full path: {os.path.abspath(temp_dir)}")

@app.post("/patients")
def add_patient(patient: Patient):
    """
    Adds a new patient to the database.

    Args:
        patient (Patient): Patient data (name, date of birth, gender).

    Returns:
        dict: A message indicating success or that the patient already exists.
    """

    response = patient_info.query(
        IndexName='info',
        KeyConditionExpression=boto3.dynamodb.conditions.Key('name').eq(patient.name) & 
                                boto3.dynamodb.conditions.Key('dob').eq(patient.dob)
    )

    if response['Items']:
        id = response['Items'][0]['id']  
        return {"message": "Patient already exists", "patient_id": id}    
    else:
        patient.id = int(time.time())
        patient_info.put_item(Item=patient.model_dump())
        return {"message": "Patient added successfully"}


@app.get("/patients")
def get_patient(name: str, dob: str = None):
    """
    Retrieves patient information based on name and optional date of birth.

    Args:
        name (str): The patient's name.
        dob (str, optional): The patient's date of birth (YYYY-MM-DD).

    Returns:
        list: A list of matching patient records.
    """
    if dob:
        response = patient_info.query(
            IndexName='info',  
            KeyConditionExpression=Key('name').eq(name)&Key('dob').eq(dob)  
        )
    else:
        response = patient_info.query(
            IndexName='info',
            KeyConditionExpression=Key('name').eq(name) 
        )

    return response['Items'] if 'Items' in response else []


@app.put("/patients/{patient_id}")
def modify_patient(patient_id: int, patient: Patient):
    """
    Updates the information of an existing patient.

    Args:
        patient_id (int): The unique ID of the patient.
        patient (Patient): The updated patient data.

    Returns:
        dict: A message indicating successful update.
    """
    patient_info.update_item(
        Key={'id': patient_id},
        UpdateExpression="set #n = :n, dob = :d, gender = :g",
        ExpressionAttributeNames={'#n': 'name'},
        ExpressionAttributeValues={
            ':n': patient.name,
            ':d': patient.dob,
            ':g': patient.gender
        },
    )
    return {"message": "Patient updated successfully"}


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int):
    """
    Deletes a patient from the database.

    Args:
        patient_id (int): The unique ID of the patient to delete.

    Returns:
        dict: A message indicating successful deletion.
    """
    patient_info.delete_item(Key={'id': patient_id})
    return {"message": "Patient deleted successfully"}


@app.get("/patients/latest")
def get_latest_patients(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Number of patients per page")
):
    now = int(time.time())
    yesterday = now - 86400 * 100

    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    query = f"""
    SELECT * FROM "patient_info" 
    WHERE id BETWEEN {yesterday} AND {now}
    """

    response = dynamodb_client.execute_statement(Statement=query)
    metadata = []

    for item in response['Items']:
        patient = {k: deserializer.deserialize(v) for k, v in item.items()}
        patient['id'] = int(patient['id'])
        metadata.append(patient)

    metadata.sort(key=lambda patient: patient['id'], reverse=True)

    # Slice the results based on pagination parameters
    paginated_metadata = metadata[start_index:end_index]

    # Check if there's a next page
    has_next_page = len(metadata) > end_index

    return {
        "patients": paginated_metadata,
        "page": page,
        "per_page": per_page,
        "has_next_page": has_next_page
    }




@app.post("/images/upload")
def upload(
    patient_id: int, 
    file: UploadFile = File(...)
):
    """
    Uploads an image to S3 and stores its information in the database.

    Args:
        patient_id (int): The ID of the patient the image is associated with.
        file (UploadFile): The image file to upload.

    Returns:
        dict: A message indicating success.
    """
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    nii_file = file.filename

    s3_key = f"ori/{nii_file}"
    # s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
    print(s3_key)
    item = Fsl(id=patient_id, name=nii_file)
    fsl.put_item(Item=item.model_dump())

    return {"message": "Upload and DB update successful!"}


@app.get("/images/{patient_id}")
def get_images_info(patient_id: int, date: str = None):
    base_query = f"SELECT * FROM \"fsl\" WHERE id = {patient_id}"

    if date:
        date_obj = datetime.strptime(date, "%Y-%m-%d")  
        start_of_day = int(date_obj.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        end_of_day = int(date_obj.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp())
        
        query = f"{base_query} AND day_upload BETWEEN {start_of_day} AND {end_of_day}"
    else:
        query = base_query

    response = dynamodb_client.execute_statement(
        Statement=query,
        Limit=20
    )

    metadata = []
    for item in response['Items']:
        image = {k: deserializer.deserialize(v) for k, v in item.items()}
        metadata.append(image)
    return metadata if 'Items' in response else []


@app.get("/images")
def get_latest_images():
    """
    Fetches information about the latest images uploaded to the system.

    Returns:
        list: A list of metadata for the latest images.
    """
    now = int(time.time())

    response = dynamodb_client.execute_statement(
        Statement=f"""SELECT * FROM "fsl"."img" WHERE day_upload <= {now}""",
        Limit=20
    )
    metadata = []
    for item in response['Items']:
        image = {k: deserializer.deserialize(v) for k, v in item.items()}
        metadata.append(image)

    return metadata if 'Items' in response else []


@app.get("/images/{image_name}/types/{type_img}")
def get_image_for_visualization(image_name: str, type_img: bool):
    """
    Downloads an image from S3 for visualization.

    Args:
        image_name (str): The name of the image file.
        type_img (bool): True for FSL image, False for original image.

    Returns:
        FileResponse: The image file as a response.
    """
    local_path = f"/tmp/{image_name}"
    print(local_path)
    s3_key = f"ori/{image_name}" if not type_img else f"fsl/{image_name}.gz"
    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
    return FileResponse(local_path, media_type='application/octet-stream', filename=image_name)

handler = Mangum(app)
