from flask import Flask,jsonify,request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route('/predict',methods = ["POST"])
def predict():
    data = request.json
    news = data['news']

    transformed_news = vectorizer.transform([news])
    prediction = model.predict(transformed_news)[0]

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug = True)
