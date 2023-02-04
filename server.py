from flask import Flask, request, render_template, jsonify
import transformers as hf

app = Flask(__name__)
tokenizer = hf.AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = hf.AutoModelForSequenceClassification.from_pretrained("./model/checkpoint-56")
pipeline = hf.pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
tokenizer_kwargs = {"padding": "max_length", "truncation": True}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        return "Bad Request: Missing text key in request data", 400
    text = data["text"]
    pred = pipeline(text, **tokenizer_kwargs)[0]
    sentiment = "Negative" if pred["label"] == "LABEL_0" else "Positive"
    score = f"{pred['score']:.4f}"
    return jsonify({"sentiment": sentiment, "score": score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345, debug=True)
