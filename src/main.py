from src.train import find_lowest_rmse,read_and_clean_data,make_model,plot_to_compare_models,find_lowest_rmse,run_best_model,plot_actual_vs_pred
from src.read_data import pull_data 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from dotenv import load_dotenv
import os
from mlflow.tracking import MlflowClient
import mlflow
import openmeteo_requests
from src.inference import pull_data_for_inference
load_dotenv()  

# Point MLflow to your SQLite database
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#set (or create) an experiment
EXPERIMENT_NAME = "temperature_model_experiment1"
mlflow.set_experiment(EXPERIMENT_NAME)
#client
client = MlflowClient()

# Check or create experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

api_path=os.getenv("API_URL")
#rename the name of the "run_name", everytime you run the main file 
def main():
    with mlflow.start_run(run_name="All_Models_Run", experiment_id=experiment_id):
        data_path=pull_data(api_path,'all',"2010-01-01") #pulls all historical data
        X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test=read_and_clean_data(data_path,'2024-12-31 00:00:00+0000','2025-01-01 00:00:00+0000')
        df_models_comp = pd.DataFrame(columns=["model_name", "rmse_train", "rmse_test"])
        make_model(X_train_scaled, X_test_scaled, y_train, y_test, LinearRegression(), 'LinearRegression',df_models_comp)
        make_model(X_train_scaled, X_test_scaled, y_train, y_test, Ridge(), 'Ridge',df_models_comp)
        make_model(X_train_scaled, X_test_scaled, y_train, y_test, Lasso(), 'Lasso',df_models_comp)
        make_model(X_train_scaled, X_test_scaled, y_train, y_test, ElasticNet(), 'ElasticNet',df_models_comp)
        make_model(X_train, X_test, y_train, y_test, RandomForestRegressor(), 'RandomForest',df_models_comp)
        make_model(X_train, X_test, y_train, y_test, xgb.XGBRegressor(), 'XGBoost',df_models_comp)
        plot_to_compare_models(df_models_comp)
        name_of_best_model=find_lowest_rmse(df_models_comp)
        y_pred,rmse_test=run_best_model(name_of_best_model,X_train, y_train,X_test,y_test,X_train_scaled,X_test_scaled)
        plot_actual_vs_pred(y_test,y_pred,X_test,name_of_best_model) 
        pull_data_for_inference(api_path,'XGBoost')  

if __name__ == "__main__":
    main()

#  mlflow ui --backend-store-uri sqlite:///mlflow.db   