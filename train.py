import mlflow
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import optuna
import random
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go
seed_value = 42
from utils import setup_logger
import datetime
from datetime import date
my_logger = setup_logger('my_temperature_logger', 'logs_for_ml.log')
my_logger.info(f"{datetime.datetime.now()}:Initialising the train.py script connection")


def read_and_clean_data(path,train_upper_date,test_lower_date):
    """
    Define a function to read in the data, clean, and split into train and test,
    and also scale the data for linear models
    """
    data=pd.read_csv(path).drop(columns='Unnamed: 0')
    data.date=pd.to_datetime(data.date)
    location_map={'Pune':1,'Mississauga':2}
    data.location=data.location.map(location_map)
    train_data=data.query(f'date<="{train_upper_date}"') #train_upper_date
    test_data=data.query(f'date>="{test_lower_date}"') #test_lower_date
    train_data.set_index('date',inplace=True)
    test_data.set_index('date',inplace=True)
    my_logger.info(f"{datetime.datetime.now()}:Testmin Min max date{test_data.index.min(),test_data.index.max()}")
    my_logger.info(f"{datetime.datetime.now()}:Train Min  max date{train_data.index.min(),train_data.index.max()}")
    my_logger.info(f"{datetime.datetime.now()}:Test Shape {test_data.shape}")
    my_logger.info(f"{datetime.datetime.now()}:Train Shape {train_data.shape}")
    X_train=train_data.drop(columns='temperature_2m')
    y_train=train_data['temperature_2m']
    X_test=test_data.drop(columns='temperature_2m')
    y_test=test_data['temperature_2m']
    scaler=MinMaxScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    my_logger.info(f"X_train Shape:{X_train.shape}")
    my_logger.info(f"X_test Shape:{X_test.shape}")
    my_logger.info(f"X_train_scaled Shape:{X_train_scaled.shape}")
    my_logger.info(f"X_test_scaled Shape:{X_test_scaled.shape}")
    my_logger.info(f"y_train Shape:{y_train.shape}")
    my_logger.info(f"y_test Shape:{y_test.shape}")
    my_logger.info(f"Type X train Shape:{type(X_train)}")
    #log data to mlflow-test, train, data
    mlflow.log_input(mlflow.data.from_pandas(X_train), context="training_features")
    mlflow.log_input(mlflow.data.from_pandas(X_test), context="testing_features")
    mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_train)), context="Y train")
    mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame(y_test)), context="Y test")
    my_logger.info(f"{datetime.datetime.now()}:Data Cleaned and Split")
    return X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test

def make_model(X_train,X_test,y_train,y_test,model,model_name,df_models_comp):
      
    """
    A function to create multiple models to compare performance 
    """
    
    model.fit(X_train,y_train)
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)
    rmse_train=np.sqrt(mean_squared_error(y_train,y_train_pred))
    rmse_test=np.sqrt(mean_squared_error(y_test,y_test_pred))
    df_models_comp.loc[len(df_models_comp.index)] = [model_name, rmse_train, rmse_test]
    # === Log trials to MLflow ===
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
    my_logger.info(f"{datetime.datetime.now()}:Ran {model_name} models to compare performance")
    return rmse_train,rmse_test

def plot_to_compare_models(df_models_comp):
    """
    Compare model performance Graphically
    """
    fig = go.Figure(data=[
    go.Bar(name='rmse_train', x=df_models_comp.model_name, y=df_models_comp.rmse_train),
    go.Bar(name='rmse_test', x=df_models_comp.model_name, y=df_models_comp.rmse_test)])
    fig.update_layout(template='plotly_dark', title='RMSE for train and test', title_x=0.5)
    my_logger.info(f"{datetime.datetime.now()}:plot_to_compare_models")
 


def find_lowest_rmse(df_models_comp):
    """
    Finds the model with the lowest RMSE Train
    """
    my_logger.info(f"{datetime.datetime.now()}:Model w Lowest TEST RMSE:{df_models_comp.sort_values(by=['rmse_test','rmse_train']).iloc[0]['model_name']}")
    my_logger.info(f"{datetime.datetime.now()}:Value Lowest RMSE:{df_models_comp.sort_values(by=['rmse_test','rmse_train']).iloc[0]['rmse_test']}")
    return df_models_comp.sort_values(by=['rmse_test','rmse_train']).iloc[0]['model_name']
    

def objective(trial,X_train, X_test, y_train, y_test):
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    train_x, valid_x, train_y, valid_y = X_train, X_test, y_train, y_test
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": 'reg:squarederror',  # Regression objective
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "random_state": seed_value,  
        "seed": seed_value 
    }

    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # train
    bst = xgb.train(param, dtrain)
    # preds
    preds = bst.predict(dvalid)
    # eval the predictions using RMSE
    rmse = np.sqrt(mean_squared_error(valid_y, preds))  # Compute RMSE
    return rmse
   
def run_best_model(name_of_best_model,X_train, y_train,X_test,y_test,X_train_scaled,X_test_scaled):
    if name_of_best_model=='XGBoost':
        # Wrap  objective to include the data
        objective_with_data = lambda trial: objective(trial, X_train, X_test, y_train, y_test)
        # set seed for Optuna
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed_value))
        study.optimize(objective_with_data, n_trials=10, timeout=600)  # Increase for more trials
        best_params = study.best_params
        best_RMSE = study.best_value
        print(f"Best Hyperparameters : {best_params}")
        print(f"Best RMSE : {best_RMSE:.4f}")
        #use the best params to create a final model 
        best_model=xgb.XGBRegressor(**study.best_params,random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        error = np.sqrt(mean_squared_error(y_test, y_pred))
        with open(f'./models/{name_of_best_model}.pkl', 'wb') as model_file:                  
            pickle.dump(best_model, model_file)
        #log as an artifact in mlflow
        mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="xgboost_reg_model",
        registered_model_name="XGB model") 
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} IS RUN AND SAVED IN MODELS DIR")      
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} has RMSE {error}")      
        return y_pred,error
    elif name_of_best_model=='LinearRegression':
        model=LinearRegression()
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
        rmse_test=np.sqrt(mean_squared_error(y_test,y_pred))
        with open(f'./models/{name_of_best_model}.pkl', 'wb') as model_file:                  
            pickle.dump(model, model_file)
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} IS RUN AND SAVED IN MODELS DIR")
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} has RMSE {rmse_test}")      
        return y_pred,rmse_test
    elif name_of_best_model=='Ridge':
        model=Ridge()
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
        rmse_test=np.sqrt(mean_squared_error(y_test,y_pred))
        with open(f'./models/{name_of_best_model}.pkl', 'wb') as model_file:                  
            pickle.dump(model, model_file)
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} IS RUN AND SAVED IN MODELS DIR")
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} has RMSE {rmse_test}")      
        return y_pred,rmse_test

    elif name_of_best_model=='RandomForest':
        model=RandomForestRegressor()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        rmse_test=np.sqrt(mean_squared_error(y_test,y_pred))
        with open(f'./models/{name_of_best_model}.pkl', 'wb') as model_file:                  
            pickle.dump(model, model_file)
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} IS RUN AND SAVED IN MODELS DIR")
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} has RMSE {rmse_test}")      
        return y_pred,rmse_test        
    elif name_of_best_model=='Lasso':
        model=Lasso()
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
        rmse_test=np.sqrt(mean_squared_error(y_test,y_pred))
        with open(f'./models/{name_of_best_model}.pkl', 'wb') as model_file:                  
            pickle.dump(model, model_file)
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} has RMSE {rmse_test}")      
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} IS RUN AND SAVED IN MODELS DIR")
        return y_pred,rmse_test
  
    else:
        model=ElasticNet()
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
        rmse_test=np.sqrt(mean_squared_error(y_test,y_pred))
        with open(f'./models/{name_of_best_model}.pkl', 'wb') as model_file:                  
            pickle.dump(model, model_file)
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} has RMSE {rmse_test}")      
        my_logger.info(f"{datetime.datetime.now()}:{name_of_best_model} IS RUN AND SAVED IN MODELS DIR")
        return y_pred,rmse_test
   
def plot_actual_vs_pred(y_test,y_pred,X_test,name_of_best_model):
    actual_vs_pred_test=pd.concat([y_test.reset_index(), pd.DataFrame(y_pred)], axis=1)
    actual_vs_pred_test.columns=['date', 'actual_temp', 'pred_temp']
    actual_vs_pred_test_w_location=pd.concat([actual_vs_pred_test,pd.DataFrame(X_test.location).reset_index(drop=True)],axis=1)
    #log this in mlflow
    mlflow.log_input(mlflow.data.from_pandas(actual_vs_pred_test_w_location), context="actual vs pred df")
    for i in actual_vs_pred_test_w_location.location.unique():
        filt_data=actual_vs_pred_test_w_location.query(f"location=={i}")
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=filt_data['date'], 
                                    y=filt_data['actual_temp'], 
                                    mode='lines', 
                                    name=f'Actual Temp Value (Region {i})'))

        # Plot INSTENSITY_VALUE as the second line
        fig.add_trace(go.Scatter(x=filt_data['date'], 
                                    y=filt_data['pred_temp'], 
                                    mode='lines', 
                                    name=f'Predicted Temp (Region {i})'))

        fig.update_layout(
            title=f'temp(Actual vs predicted) by  Time for Region {i}',
            xaxis_title='Reading Timestamp',
            yaxis_title='Value',
            xaxis=dict(tickformat='%Y-%m-%d %H:%M:%S'),
        )

        fig.show()
        plot_path = f"actual_vs_predicted_{i}.png"
        fig.savefig(plot_path)

        # Log the plot as an artifact
        mlflow.log_artifact(plot_path, "plots") # "plots" is an optional subdirectory within the artifacts


    plt.close(fig) # Close the plot to free up memory
    my_logger.info(f"{datetime.datetime.now()}:Actual vs Pred PLOT SAVED")

if __name__=="__main__":
    read_and_clean_data('./data/temp_data_all_2025-10-23.csv','2024-12-31 00:00:00+0000','2025-01-01 00:00:00+0000')