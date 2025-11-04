import os 
from google.cloud import aiplatform
import pickle
from google.cloud import storage
from dotenv import load_dotenv
load_dotenv()

#initialise the project AND OTHER PARAMS
project_id=os.getenv('PROJECT_ID')
aiplatform.init(project=project_id, location="us-central1")


def get_model():

    model = aiplatform.Model(model_name=f"projects/{project_id}/locations/us-central1/models/7195550438328893440")

    print("Model loaded:", model.resource_name)

    print('model is loaded')


if __name__=="__main__":
    get_model()