# IMDB Sentiment Classification with BERT

## 📌 Overview
This project implements binary sentiment classification (positive/negative) on the IMDB movie reviews dataset using the pre-trained BERT (bert-base-uncased) language model. By fine-tuning BERT on real-world sentiment data, we achieve high classification accuracy with minimal preprocessing and feature engineering.

## ❓ Why Choose BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model pre-trained on a large English corpus. Unlike traditional word embeddings like Word2Vec or GloVe, BERT captures context bidirectionally and supports fine-tuning for downstream tasks such as sentiment classification.

In this project, we use BERT to:
- Automatically learn semantic representations of reviews.
- Minimize the need for manual feature engineering.
- Achieve excellent performance in binary sentiment classification.

## 📂 Dataset
- **Source**: IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Size**: 50,000 reviews (25k positive, 25k negative)
- **Format**: CSV with columns: review and sentiment

### Preprocessing:
- Null values are filled with 'none'.
- Sentiment labels are mapped to integers: 'positive' → 1, 'negative' → 0.

### Data Split:
- 60% for training.
- 20% for validation.
- 20% for testing (held out for final evaluation).

## ⚙️ Model Configuration
- **Model**: bert-base-uncased from HuggingFace Transformers
- **Architecture**:
  - BERT encoder
  - Dropout (p = 0.3)
  - Linear layer (output: 1 unit)
- **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup and decay
- **Max Sequence Length**: 512 tokens

## 🧪 Training Details
- **Batch Size**: 8 (train), 4 (validation/test)
- **Epochs**: 3
- **Device**: GPU (CUDA)
- **Tokenizer**: BERT tokenizer with do_lower_case=True

### Model performance across epochs (on validation set):
- **Epoch 1**: Accuracy = 93.12%
- **Epoch 2**: Accuracy = 94.05%
- **Epoch 3**: Accuracy = 94.34%

## 🧾 Test Evaluation
The final model is tested on the held-out 20% test set:

- **Test Accuracy**: 94.49%