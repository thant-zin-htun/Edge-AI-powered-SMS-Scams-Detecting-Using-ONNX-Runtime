# Edge AI-powered Japanese SMS Scam Detection Using ONNX Runtime

This project implements an efficient Edge AI solution for detecting scam messages in Japanese SMS texts using optimized machine learning models deployed with ONNX Runtime.

## 📋 Overview

This system is designed to identify potentially fraudulent SMS messages in Japanese language contexts. By leveraging multiple machine learning models (Naive Bayes, Random Forest, Logistic Regression, and SVM) and selecting the highest performing one, the system provides accurate scam detection while being lightweight enough to run on edge devices.

## ✨ Features

- **Multi-model approach**: Evaluates Naive Bayes, Random Forest, Logistic Regression, and SVM models to select the highest performers
- **Optimized for edge deployment**: Utilizes ONNX Runtime for efficient inference on resource-constrained devices
- **Japanese language support**: Specifically designed for processing and analyzing Japanese text messages
- **High accuracy detection**: Selects the best performing models to ensure optimal scam identification
- **Low latency**: Engineered for quick response times, essential for real-time message screening

## 🔧 Technologies

- Python
- ONNX Runtime
- Scikit-learn (for model training)
- MeCab (for Japanese tokenization)
- NumPy/Pandas (for data processing)

## 🔄 Workflow

The project follows this workflow for spam detection:

```
┌───────────────┐
│      SMS      │
│    Dataset    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│     Data      │
│ Preprocessing │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Feature     │
│  Extraction   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Machine     │
│   Learning    │
│    Model      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Model      │
│  Evaluation   │
└───────┬───────┘
        │
        ▼
        ◆
┌───────────────┐
│  Prediction   │
│  with Actual  │
│     Data      │
└───────┬───────┘
    ┌───┴───┐
    ▼       ▼
┌───────┐ ┌───────┐
│ Spam  │ │  Ham  │
└───────┘ └───────┘
```

## 📊 Model Performance

| Model | Accuracy |  Latency  |
|-------|----------|-----------|
| Naive Bayes | 99.94% | 0.01s |
| Logistic Regression | 100% | 0.02s |
| Random Forest | 100% | 0.64s |
| SVM | 100% | 5s |


## 📚 How It Works

1. **Text Preprocessing**: Japanese SMS messages are tokenized using MeCab and transformed into feature vectors
2. **Model Training**: Multiple classifiers (Naive Bayes, Random Forest, Logistic Regression, SVM) are trained on the dataset
3. **Model Selection**: The highest performing models based on accuracy are selected
4. **ONNX Conversion**: The selected models are converted to ONNX format for optimized inference
5. **Edge Deployment**: The ONNX models are deployed using ONNX Runtime for efficient execution on edge devices

