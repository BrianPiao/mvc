from flask import Flask, jsonify, request
from main import get_pred

app = Flask(__name__)

@app.route("/")
def func():
    return "heya"

@app.route("/predict-digit" , methods = ["POST"])
def predictData():
    image = request.files.get("digit")
    p = get_pred(image)
    return jsonify({
        "prediction":p
    })
    
if __name__ == "__main__":
    app.run(debug = True)