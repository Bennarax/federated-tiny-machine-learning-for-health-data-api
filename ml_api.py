from flask import Flask, jsonify, request
from configparser import ConfigParser

import pandas as pd
import numpy as np
import traceback
import joblib
import json
import os


def get_model_info(ini_path=None):
    if ini_path is None:
        ini_path = r"ml_api_config.ini"

    configur = ConfigParser()
    configur.read(ini_path)

    model_folder = configur.get('folders', 'model_folder')

    diseases = configur.get('diseases', 'list_diseases')
    diseases = diseases.split(" ; ")

    list_diseases_dataset = configur.get('diseases', 'list_diseases_dataset')
    list_diseases_dataset = list_diseases_dataset.split(" ; ")

    dict_diseases = {}
    for i in range(len(diseases)):
        disease_dataset = list_diseases_dataset[i]

        nb_pred = int(configur.get(disease_dataset, 'nb_pred')) if configur.get(disease_dataset, 'nb_pred') else 10
        SMOTE = configur.get(disease_dataset, 'use_SMOTE') in ["1", "True", "true"]
        PCA = configur.get(disease_dataset, 'keep_most_important_components') in ["1", "True", "true"]
        model_type = configur.get(disease_dataset, 'model')

        SMOTE_file_name = '_SMOTE' if SMOTE else ''
        keep_file_name = '_pca' if PCA else ''

        model_name = model_type + SMOTE_file_name + keep_file_name
        model_file_name = model_name + ".pkl"
        model_file_path = os.path.join(model_folder, disease_dataset, model_file_name)

        columns_file_name = "models_columns" + keep_file_name + ".pkl"
        columns_file_path = os.path.join(model_folder, disease_dataset, columns_file_name)

        precisions_file_path = os.path.join(model_folder, disease_dataset, "models_precisions.json")
        with open(precisions_file_path, 'r') as json_file:
            models_precisions = json.load(json_file)

        dict_diseases[diseases[i]] = {"model": joblib.load(model_file_path),
                                      "model_columns": joblib.load(columns_file_path),
                                      "model_precision": models_precisions[model_name],
                                      "nb_pred": nb_pred}

    return dict_diseases


dict_diseases = get_model_info()

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def main_predict():
    if dict_diseases:
        try:
            data_request = str(request.data)[2:-1]
            list_data = [d for d in data_request.split(";")]

            disease = list_data[0]

            model = dict_diseases[disease]["model"]
            model_columns = dict_diseases[disease]["model_columns"]
            model_precision = dict_diseases[disease]["model_precision"]
            nb_pred = dict_diseases[disease]["nb_pred"]

            data = {model_columns[i]: [float(list_data[i+1])] for i in range(len(model_columns))}
            query = pd.DataFrame(data)

            prediction, trust_rate = predict(query, model, nb_pred, model_precision)
            result = "Result : negative." if prediction == 0 else "Result : positive !"
            result += " (trust rate = " + str(trust_rate) + ")"

            return result

        except:
            disease = disease or "! error when get disease !"
            print("Error when try predict with the model of " + disease + " .\n", traceback.format_exc())
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Error when try load models.')
        return 'No model here to useNo model here to use'


def predict(data, model, nb_pred, model_precision):
    list_predictions = list()
    for _ in range(nb_pred):
        prediction = np.round(model.predict(data)[0])
        list_predictions.append(prediction)

    sum_predictions = sum(list_predictions)
    sum_predictions /= nb_pred

    prediction = round(sum_predictions)

    trust_rate = 1 - sum_predictions if prediction == 0 else sum_predictions
    trust_rate = round(trust_rate * model_precision * 100, 2)

    return prediction, trust_rate


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
