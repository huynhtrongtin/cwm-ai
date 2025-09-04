import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

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
# Test trực tiếp
# ----------------------------
if __name__ == "__main__":
    while True:
        text = input("Nhập câu để AI nhận nguyên liệu (hoặc 'exit' để thoát): ")
        if text.lower() == "exit":
            break
        result = extract_ingredients(text)
        print("Nguyên liệu:", result)
