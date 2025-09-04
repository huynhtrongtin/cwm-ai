from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import gdown
import zipfile

# ----------------------------
# Thiết lập model path
# ----------------------------
model_path = "model_AI/ingredient-model"

# Nếu chưa có model, tải từ Google Drive
if not os.path.exists(model_path):
    os.makedirs("model_AI", exist_ok=True)
    # Thay <file_id> bằng ID public của zip trên Drive
    url = "https://drive.google.com/drive/folders/1tA2Tt2z3nDs80xSfOLTR-2htvVXGi0hF"
    gdown.download(url, "model.zip", quiet=False)
    # Giải nén
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall(model_path)

# ----------------------------
# Load model + tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

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

# ----------------------------
# Chạy Flask server
# ----------------------------
if __name__ == "__main__":
    # Railway cung cấp PORT qua biến môi trường
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
