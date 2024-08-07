import boto3, time, os, random

from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from model_utils import load_and_preprocess_image, predict, model

from mangum import Mangum
from boto3.dynamodb.conditions import Key
from pydantic import BaseModel, Field
from typing import Optional
from boto3.dynamodb.types import TypeDeserializer

S3_BUCKET_NAME=os.getenv("S3_BUCKET_NAME")
TEMP_DIR = '/tmp'

s3_client = boto3.client('s3') 
dynamodb_resource = boto3.resource('dynamodb')
dynamodb_client = boto3.client('dynamodb')

patient_info = dynamodb_resource.Table('patient_info')
fsl = dynamodb_resource.Table('fsl')

deserializer = TypeDeserializer()
app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )
handler = Mangum(app)

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


@app.post("/patients")
def add_patient(patient: Patient):
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
        patient_info.put_item(Item=patient.dict())
        return {"message": "Patient added successfully"}


@app.get("/patients")
def get_patient(name: str, dob: str = None):
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
    patient_info.delete_item(Key={'id': patient_id})
    fsl.delete_item(Key={'id': patient_id})
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
        "has_next_page": has_next_page
    }


@app.post("/images/upload")
def upload(patient_id: int, file_name: str ):

    s3_key = f"ori/{file_name}"
    presigned_url = s3_client.generate_presigned_url(
        ClientMethod='put_object',
        Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
        ExpiresIn=300  # Adjust expiration time as needed
    )    
    item = Fsl(id=patient_id, name=file_name)
    fsl.put_item(Item=item.dict())

    return presigned_url


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


@app.get("/images/{image_name}/vilz")
def get_image_for_visualization(image_name: str):

    s3_key = f"fsl/{image_name}" if image_name.endswith(".gz") else f"fsl/{image_name}.gz"
    presigned_url = s3_client.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
        ExpiresIn=300
    )
    return presigned_url

@app.post("/patients/{patient_id}/images/{image_name}/at/{day_upload}")
def predict_and_update(patient_id: int, image_name: str, day_upload: int):

    s3_key = f"fsl/{image_name}.gz"
    
    local_file_path = os.path.join(TEMP_DIR, image_name)
    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
    
    image = load_and_preprocess_image(local_file_path)
    
    predicted_class, _ = predict(model, image)
    
    fsl.update_item(
        Key={'id': patient_id, 'day_upload': day_upload},
        UpdateExpression="SET predict = :p",
        ExpressionAttributeValues={':p': predicted_class}
    )
    
    os.remove(local_file_path)
    
    return {
        "message": f"Prediction ({predicted_class}) and update successful for image {image_name}",
    }


@app.post("/patients/{patient_id}/day/{day_upload}/note")
def update_note(patient_id: int, day_upload: int, note: str):
    fsl.update_item(
        Key={'id': patient_id, 'day_upload': day_upload},
        UpdateExpression="SET note = :n",
        ExpressionAttributeValues={':n': note}
    )

    return {"message": f"Note update successful for patient {patient_id} on day {day_upload}"}

