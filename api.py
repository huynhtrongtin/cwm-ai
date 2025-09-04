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
model_dir = "ingredient-model"
file_id = "YOUR_FILE_ID"   # <-- thay bằng ID file model.zip trong Drive
output = f"{model_dir}.zip"

if not os.path.exists(model_dir):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Model extracted!")
# ----------------------------
# Load model + tokenizer
# ----------------------------
model_name = "ingredient-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
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
    app.run(port=5000, debug=True)
