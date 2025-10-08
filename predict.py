# predict.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

# Load model and tokenizer from Hugging Face Hub
model_name = "me-r/esg-multilabel-classification" 
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load label encoder
with open("esg_label_encoder.json", 'r') as f:
    class_names = json.load(f)['classes']

# Prepare your text."
text = "Our board has introduced stricter ethical guidelines, improved transparency in business operations, and enhanced diversity across executive roles."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).numpy()[0]

# Get predicted labels (using threshold of 0.5)
predicted_labels = [class_names[i] for i, prob in enumerate(probs) if prob > 0.5]
print(f"Predicted ESG Categories: {predicted_labels}")