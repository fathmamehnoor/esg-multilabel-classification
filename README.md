# 🌱 ESG Multi-Label Classification with BERT

This project fine-tunes a BERT model to classify text from sustainability reports into ESG (Environmental, Social, Governance) categories. It uses PyTorch and Hugging Face Transformers and supports multi-label classification.

Achieved an **F1 Score of 0.9501** on validation data.

---

## 🧠 Features

- Fine-tunes `bert-base-uncased` for multi-label ESG classification
- Predicts relevant ESG categories from textual input
- Uses sigmoid activation for multi-label outputs
- Includes training and prediction scripts
- Stores ESG label classes and enables inference using saved models

---

## 📁 Project Structure

```
├── predict.py              # Load trained model and run predictions
├── esg_label_encoder.json  # Saved label encoder (for inference)
├── requirements.txt        # Required packages
└── README.md               # This file
```

---

## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```



## 🔍 Inference

You can run predictions on new text like this:

```bash
python predict.py
```

Example Prediction:
Input Text: Our company is investing in solar energy and reducing emissions across all facilities.
Predicted ESG Categories: ['Energy_Management']
Top Probabilities:
  Energy_Management: 0.9647
  Product_Design_And_Lifecycle_Management: 0.0049
  Waste_And_Hazardous_Materials_Management: 0.0046


You can modify example_text in predict.py with your own input.

## 🧪 Model

- Base Model: bert-base-uncased
- Task: Multi-label classification
- Evaluation Metric: Micro F1 Score
- Best F1: 0.9501
