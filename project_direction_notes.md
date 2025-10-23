#the project goes this way
#creat a git repo -Done
#create a venve-Done
#add gitignore file for csvs -Done
#write code to pull in hourly historical temperature data and store in a csv/cloud bucket. make sure error handling and logging is in place. Maybe create a 
utils file to pull in the logger. -Done
#use this file to train a model to predict temp for each hour  jupyter
#make sure you use logging, -done
#and log experiments with mlflow  TBD
#once this is done, create an inference pipelinee/script that will read hourly data, and give out predictions
#this should shouw viz in the dashbaord -maybe use fastapi 
#compare the preds to actuals as well(this comes from the API)
#WHAT CAN BE DONE TO PARALLELIZE THE STEPS IF THERES TOO MUCH DATA?
#CAN WE DECOUPLE USING A FEATURE STORE 
#GIT HUB ACTIONS-THINK ABOUT HOW WE CAN DEPLOY WITH GA
#add a flow to connect all the training steps, similar for the inference 
pipeline. Trigger train and retrain when needed
#do the above locally first 
##Step 2:
#dockerise this(ask if you need to dockerise the whole app or just inference script)
#use gcp to schedule hourly runs 

#FOR LATER 
#TRACK DATA DRIFT AND PREDICTION DRIFT 
#set up alerts if something fails 
#mlflow for tracking
    