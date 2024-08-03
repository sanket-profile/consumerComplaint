import warnings
warnings.filterwarnings('ignore')

import os

from flask import Flask
from flask import request, render_template, url_for, redirect,jsonify
import requests

from src.pipelines.prediction_pipeline import predictionPipeline

application = Flask(__name__)
app = application


@app.route("/",methods=["GET"])
def home():
    return redirect(url_for('predict'))

@app.route("/predict",methods=["GET","POST"])
def predict():
    if(request.method == "GET"):
        return render_template("predict.html")
    else:
        tfidf = os.path.join(os.getcwd(),"Artifacts","tfidf.pkl")
        model = os.path.join(os.getcwd(),"Artifacts","logisticModel.pkl")
        le = os.path.join(os.getcwd(),"Artifacts","le.pkl")
        X = request.form.get("inputText")
        print(X)
        predict_pipeline = predictionPipeline()
        product = predict_pipeline.predict(X=str(X),tfidfPath=tfidf,modelPath=model,lePath=le)
     
        # Return predictions and image URLs as JSON response
        return render_template('predict.html', results=product)






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)