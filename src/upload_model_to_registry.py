import os 
from google.cloud import aiplatform
import pickle
from google.cloud import storage
from dotenv import load_dotenv
load_dotenv()
#had to run gcloud auth application-default login to configure 


#initialise the project AND OTHER PARAMS
project_id=os.getenv('PROJECT_ID')
aiplatform.init(project=project_id, location="us-central1")
model_path='XGBoost.pkl'
BUCKET=os.getenv('BUCKET')


#upload to gcs 
def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)
  print('File {} uploaded to {}.'.format(
      source_file_name,
      destination_blob_name))  

#write to model registry
def load_to_registry():
    """loads model to gcp model registry 
    """
    model = aiplatform.Model.upload(
        display_name="xgboost-3-1-0-model",
        artifact_uri="gs://temp_model_artifacts/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.3-1:latest",
    )
    print('Model is loaded to artifact registry in GCP')

if __name__=="__main__":
  upload_blob(BUCKET, './models/XGBoost.pkl', f'{model_path}')
  load_to_registry()

