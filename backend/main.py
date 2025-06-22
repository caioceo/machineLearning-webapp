from flask import Flask, request;
from flask_cors import CORS;
import pandas as pd;
import io
import json
from models.XGBoost import train_categorical_model

main = Flask(__name__)

CORS(main)

@main.route("/gerar", methods=["POST"])
def generator():
    target = request.form.get("target")
    file = request.files["file"]
    data = request.form.get("columns")
    columns = json.loads(data)
    
    csvFile = file.read().decode("utf-8")
    
    df = pd.read_csv(io.StringIO(csvFile))
    
    print(columns)
    print(target)
    results = train_categorical_model(df, target, columns)

    return "sucesso"
    