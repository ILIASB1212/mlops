import pandas as pd
import yaml
import os


params=yaml.safe_load(open("params.yaml"))["preproses"]
#preproses
def prossess_data(input,output):
    data=pd.read_csv(input)
    dir_name=os.path.dirname(output)
    os.makedirs(dir_name,exist_ok=True)
    data.to_csv(output,index=False)
    return f"data prossessed and saved to {output}"



if __name__=="__main__":

    input=params["input"]
    output=params["output"]
    prossess_data(input,output)