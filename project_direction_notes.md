#the project goes this way
#creat a git repo -Done
#create a venve-Done
#add gitignore file for csvs -Done
#log MLFLOW DATA FILES TRAINING VERSIONS 
#write code to pull in hourly historical temperature data and store in a csv/cloud bucket. make sure error handling and logging is in place. Maybe create a utils file to pull in the logger. -Done
#use this file to train a model to predict temp for each hour-done
#make sure you use logging, -done
#and log experiments with mlflow  Done.

#do cicd locally-today tbd 


#once this is done, create an inference pipelinee/script that will read hourly data, and give out predictions-done
#append each hourly pred to a cloud storage, or db 
#from the above, read and stream into a dashboard to compare actuals vs pred, and RMSE  
#WOULD FASTAPI BE OF ANY USE? 
#WHAT CAN BE DONE TO PARALLELIZE THE STEPS IF THERES TOO MUCH DATA?-chunk it 
#CAN WE DECOUPLE USING A FEATURE STORE 
#GIT HUB ACTIONS-THINK ABOUT HOW WE CAN DEPLOY WITH GA
#add a flow to connect all the training steps, similar for the inference 
pipeline. Trigger train and retrain when needed
#do the above locally first 
##Step 2:
#dockerise this(ask if you need to dockerise the whole app or just inference script)
#I dockerised the Inference.py, I need to figure out how to talk to gcp when dockerising!!!!
#https://cloud.google.com/docs/authentication/application-default-credentials#GAC
#use gcp to schedule hourly runs 

#FOR LATER 
#TRACK DATA DRIFT AND PREDICTION DRIFT 
#set up alerts if something fails 
#mlflow for tracking
   docker build -t temperature-inference . 
   docker run `  -v "C:\Users\Sanaaya\Desktop\temperature_data_predictor_batch\aeolus-northamerica-5d3691721c5a.json:/app/key.json" `  -e GOOGLE_APPLICATION_CREDENTIALS="/app/key.json" ` temperature-inference
   docker tag temperature-inference:latest REGION-docker.pkg.dev/PROJECT_ID/inference-repo/temperature-inference:latest

   ##creat artifact registry  repo and push the image to it 
   gcloud artifacts repositories create inference-repo --repository-format=docker --location=us-central1
aeolus-northamerica
   #push the image to the artifact registry 
   docker tag temperature-inference us-central1-docker.pkg.dev/aeolus-northamerica-ai-sandbox/inference-repo/temperature-inference:latest
   gcloud auth configure-docker us-central1-docker.pkg.dev
   docker push us-central1-docker.pkg.dev/aeolus-northamerica/inference-repo/temperature-inference:latest

#deploy the image to cloud run JOBS and Not cloud run services. services is for https 
requrest, and jobs is for batch/script runs 
gcloud run jobs create temperature-inference-job --image=us-central1-docker.pkg.dev/aeolus-northamerica/inference-repo/temperature-inference:latest 
  --region=us-central1 
  --memory=1Gi 
  --set-env-vars=GOOGLE_APPLICATION_CREDENTIALS="/app/key.json"
gcloud run jobs execute temperature-inference-job --region=us-central1
 #the above doesnt work because gc creds are not mounted on the image, unlike in the local run 
 #we need to pass the service account that has the creds 
 gcloud run jobs create temperature-inference-job 
    --image=us-central1-docker.pkg.dev/aeolus-northamerica/inference-repo/temperature-inference:latest 
    --region=us-central1 
    --service-account=sa-indeed@aeolus-northamerica.iam.gserviceaccount.com


#Notes on project 
#I first created a few functions that pulled in the data, trained and found the best model, and stored the model as a pkl 
# I then created an infenece py file that called in data to run preds on 
#I created a repo on git and made sure to push everything
#I dockerised the src file and the inference py file, using a docker file
#I also implemented a push workflow that built the docker image after pushing to git hub(github actions)
#Since all of the data retrieving+model calling+logging etc was done locally, I wanted to shift
it to gcp the cloud 
#I changed the inf py code to read the model from GCP bucket and write preds to it 
#I re dockerised my scripts and got them to work
#for docker to comm with gcp-I had to mount a drive on the container that would 
read the creds for gcp 
#I couldnt figure out how to make a build workblow for this on github actions, 
because my docker ontainer was unable to connect with my gcp project 
#I let it go 
#I then build an artifact repo on the gcp project 
#I pushed the image to the artifact registy 
#I used cloud run JOBS to create a job to run the docker image. 
#i RAN INTO AN ERROR BECAUSE in running docker run locally, i was mounting the file with -v …:/app/key.json. However on gcp, I have to use a service account to pass the creds 
