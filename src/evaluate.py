import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

from src import load_prepro



os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/ILIASB1212/mlops.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="ILIASB1212"
os.environ['MLFLOW_TRACKING_PASSWORD']="f4e52b01812a1f3ec2f179f5ea4e7e86c27200af"


def evaluate(model_path,data_path,scaler_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=data.columns[-1])
    y=data[data.columns[-1]]
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    y_pred=model.predict(X_scaled)
    acc=accuracy_score(y,y_pred)
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("mlops_project")
    with mlflow.start_run(run_name="evaluate_model"):
        mlflow.log_metric("accuracy", acc)




if __name__=="__main__":
    params=yaml.safe_load(open("params.yaml"))["train"]
    params_1=yaml.safe_load(open("params.yaml"))["preproses"]

    model_path=params["saaved_model"]
    data_path=params["input"]
    scaler_path=params_1["scaler"]
    evaluate(model_path,data_path,scaler_path)






