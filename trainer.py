from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch

# ----------------------------
# Load data
# ----------------------------
dataset = load_dataset("json", data_files="ingredients_augmented.jsonl")

# ----------------------------
# Tokenizer
# ----------------------------
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ----------------------------
# Chuyển labels string sang token-level labels
# ----------------------------
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(
        batch["text"], 
        truncation=True, 
        padding="max_length", 
        return_offsets_mapping=True
    )

    labels = []
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        word_labels = batch["labels"][i]  # danh sách nguyên liệu cho câu này
        label_ids = []

        # Chuyển từ word_labels sang nhãn token: 1 nếu token thuộc nguyên liệu, 0 nếu không
        text = batch["text"][i].lower()
        current_pos = 0
        for start, end in offsets:
            if start == end:  # padding token
                label_ids.append(-100)  # Huggingface convention để bỏ qua loss
                continue
            token_text = text[start:end]
            matched = False
            for ingredient in word_labels:
                if ingredient.lower().find(token_text) != -1:
                    matched = True
                    break
            label_ids.append(1 if matched else 0)
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    # xóa offset mapping vì không cần nữa
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs


dataset = dataset.map(tokenize_and_align_labels, batched=True)

# ----------------------------
# Model
# ----------------------------
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)  # 0=không,1=nguyên liệu

# ----------------------------
# Training
# ----------------------------
args = TrainingArguments(
    output_dir="ingredient-model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],  # tạm dùng train cho eval
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("ingredient-model")
