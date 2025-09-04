from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import gdown
import zipfile

# ----------------------------
# Tải model từ Google Drive
# ----------------------------
model_name = "/content/drive/MyDrive/app_cwm/model_AI/ingredient-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
# ----------------------------
# Load model + tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

print(f"Device set to use {device}")

# ----------------------------
# Hàm extract nguyên liệu
# ----------------------------
def extract_ingredients(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    labels = torch.argmax(logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    ingredients = []
    current = []
    for token, label in zip(tokens, labels):
        if label == 1:
            current.append(token)
        else:
            if current:
                word = tokenizer.convert_tokens_to_string(current)
                ingredients.append(word)
                current = []
    if current:
        word = tokenizer.convert_tokens_to_string(current)
        ingredients.append(word)

    # loại bỏ các PAD còn sót
    ingredients = [w for w in ingredients if "[PAD]" not in w]

    return ingredients

# ----------------------------
# Flask API
# ----------------------------
app = Flask(__name__)
CORS(app)  # enable CORS nếu cần gọi từ trình duyệt

@app.route("/api/extract", methods=["POST"])
def api_extract():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    ingredients = extract_ingredients(text)
    return jsonify({"ingredients": ingredients})

if __name__ == "__main__":
    app.run(debug=True)
