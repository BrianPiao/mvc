from flask import Flask, jsonify, request
from projA import get_pred

app = Flask(__name__)

@app.route("/")
def func():
    return "alpha"

@app.route("/predict-alphabet" , methods = ["POST"])
def predictData():
    image = request.files.get("alphabet")
    p = get_pred(image)
    return jsonify({
        "prediction":p
    })
    
if __name__ == "__main__":
    app.run(debug = True)