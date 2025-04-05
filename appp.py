import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

#  Load the trained model
with open('language_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

#  Load the correct vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

#  Debug: Check vocabulary size of loaded vectorizer
print("Vectorizer Vocabulary Size:", len(vectorizer.get_feature_names_out()))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]

        #  Transform input text using the SAME vectorizer
        text_vectorized = vectorizer.transform([text])

        #  Debug: Check shape of vectorized input
        print("Vectorized input shape:", text_vectorized.shape)

        #  Predict language
        prediction = model.predict(text_vectorized)[0]

        return render_template("index.html", prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)



