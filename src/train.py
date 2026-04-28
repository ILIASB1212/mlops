import math
import os.path
from py_compile import main
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
import mlflow
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/ILIASB1212/mlops.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="ILIASB1212"
os.environ['MLFLOW_TRACKING_PASSWORD']="f4e52b01812a1f3ec2f179f5ea4e7e86c27200af"




def hyper_parametre_tuning(x_train,y_train,params):
    rf=RandomForestClassifier()
    gs=GridSearchCV(estimator=rf,param_grid=params,cv=5)
    gs.fit(x_train,y_train)
    return gs



params=yaml.safe_load(open("params.yaml"))["train"]
scaler_file=yaml.safe_load(open("params.yaml"))["preproses"]["scaler"]

def standard_scaling(x_train):
    scaler=StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    file_name=os.path.dirname(scaler_file)
    os.makedirs(file_name,exist_ok=True)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    return x_train_scaled, scaler



def train(data_path,model_path,random_state,n_estimators,max_depth,test_size):
    data=pd.read_csv(data_path)
    X=data.drop(columns=data.columns[-1])
    y=data[data.columns[-1]]
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("mlops_project")
    

    with mlflow.start_run(run_name="train_model"):
        x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
        signature=infer_signature(x_train,y_train)

        ## Define hyperparameter grid

        x_train_scaled, scaler=standard_scaling(x_train)
        param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]}

        gs=hyper_parametre_tuning(x_train_scaled,y_train,param_grid)
        best_rf=gs.best_estimator_
        x_test_scaled = scaler.transform(x_test)
        y_pred=best_rf.predict(x_test_scaled)

        acc=accuracy_score(y_test,y_pred)
        print(f"accuracy: {acc}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("n_estimators", best_rf.n_estimators)
        mlflow.log_param("max_depth", best_rf.max_depth)
        mlflow.log_param("min_samples_split", best_rf.min_samples_split)
        mlflow.log_param("min_samples_leaf", best_rf.min_samples_leaf)
        mlflow.sklearn.log_model(best_rf, "rf_model", signature=signature)

        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")


        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(best_rf,"rf_model",registered_model_name="Best Model")
        else:
             mlflow.sklearn.log_model(best_rf, "rf_model", signature=signature)
            
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_rf, open(model_path, 'wb'))



if __name__ == "__main__":
    train(params['input' ], params['saaved_model' ],params['random_state' ],params['n_estimators'],params['max_depth' ],params['test_size' ])



