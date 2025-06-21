from flask import Flask, request;
from flask_cors import CORS;
import pandas as pd;
import io

main = Flask(__name__)

CORS(main)

@main.route("/", methods=["GET"])
def pin():
    return "jasbdba"

@main.route("/gerar", methods=["POST"])
def generator():
    target = request.form.get("target")
    file = request.files["file"]
    model = request.form.get("model")
    
    csvFile = file.read().decode("utf-8")
    
    df = pd.read_csv(io.StringIO(csvFile))
    
    cols = df.columns.tolist()
    
    print(target)
    print(cols)
    print (model)
    
    match model:
        case "LightGBM":
            print("LIGHT")
        case "XGBoost":
            print("BOOSt")

    return "sucesso"
    