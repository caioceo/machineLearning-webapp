from flask import Flask, request;
from flask_cors import CORS;
import pandas as pd;
import io
from models.XGBoost import xgboost
from models.XGBoost import train_categorical_model

main = Flask(__name__)

CORS(main)

@main.route("/gerar", methods=["POST"])
def generator():
    target = request.form.get("target")
    file = request.files["file"]
    model = request.form.get("model")
    targetValues = request.form.get("values") 
    
    csvFile = file.read().decode("utf-8")
    
    df = pd.read_csv(io.StringIO(csvFile))
    
    cols = df.columns.tolist()
    
    print(target)
    print(cols)
    print (model)
    
    match model:
        case "XGBoost":
            results = train_categorical_model(df, target, cols)

    return "sucesso"
    