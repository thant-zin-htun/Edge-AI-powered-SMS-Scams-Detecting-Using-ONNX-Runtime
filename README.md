# Edge AI-powered Japanese SMS Scam Detection Using ONNX Runtime

This project implements an efficient Edge AI solution for detecting scam messages in Japanese SMS texts using optimized machine learning models deployed with ONNX Runtime.

## 📋 Overview

This system is designed to identify potentially fraudulent SMS messages in Japanese language contexts. By leveraging multiple machine learning models (Naive Bayes, Random Forest, Logistic Regression, and SVM) and selecting the highest performing one, the system provides accurate scam detection while being lightweight enough to run on edge devices.

## ✨ Features

- **Multi-model approach**: Evaluates Naive Bayes, Random Forest, Logistic Regression, and SVM models to select the highest performer
- **Optimized for edge deployment**: Utilizes ONNX Runtime for efficient inference on resource-constrained devices
- **Japanese language support**: Specifically designed for processing and analyzing Japanese text messages
- **High accuracy detection**: Automatically selects the best performing model to ensure optimal scam identification
- **Low latency**: Engineered for quick response times, essential for real-time message screening

## 🔧 Technologies

- Python
- ONNX Runtime
- Scikit-learn (for model training)
- MeCab/Fugashi (for Japanese tokenization)
- NumPy/Pandas (for data processing)

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | X% | X% | X% | X% |
| Random Forest | X% | X% | X% | X% |
| Logistic Regression | X% | X% | X% | X% |
| SVM | X% | X% | X% | X% |

*Note: Replace X with actual performance metrics from your evaluation*

## 🚀 Getting Started

### Prerequisites

```
- Python 3.7+
- ONNX Runtime
- MeCab (for Japanese tokenization)
- Other dependencies listed in requirements.txt
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Edge-AI-powered-Japanese-SMS-Scams-Detecting-Using-ONNX-Runtime.git
cd Edge-AI-powered-Japanese-SMS-Scams-Detecting-Using-ONNX-Runtime
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install MeCab for Japanese language processing (if not included in requirements):
```bash
# For Ubuntu/Debian
sudo apt-get install mecab mecab-ipadic-utf8

# For macOS
brew install mecab mecab-ipadic
```

## 💻 Usage

### Model Training

To train and evaluate all models:

```bash
python train_models.py --data_path ./data/sms_dataset.csv
```

### Converting to ONNX Format

To convert the best performing model to ONNX format:

```bash
python convert_to_onnx.py --model_path ./models/best_model.pkl --output ./models/best_model.onnx
```

### Running the Detector

To detect scams in new messages:

```bash
python detect_scam.py --input "メッセージテキスト" --model ./models/best_model.onnx
```

Or to process a batch file:

```bash
python detect_scam.py --input_file ./data/messages.txt --output_file ./results/predictions.csv
```

## 📂 Project Structure

```
├── data/                  # Dataset files
├── models/                # Trained models and ONNX files
├── src/
│   ├── preprocessing.py   # Text preprocessing for Japanese
│   ├── train_models.py    # Model training and selection
│   ├── convert_to_onnx.py # Script to convert models to ONNX
│   ├── detect_scam.py     # Inference script
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 📚 How It Works

1. **Text Preprocessing**: Japanese SMS messages are tokenized using MeCab and transformed into feature vectors
2. **Model Training**: Multiple classifiers (Naive Bayes, Random Forest, Logistic Regression, SVM) are trained on the dataset
3. **Model Selection**: The highest performing model based on accuracy is selected
4. **ONNX Conversion**: The selected model is converted to ONNX format for optimized inference
5. **Edge Deployment**: The ONNX model is deployed using ONNX Runtime for efficient execution on edge devices

## 🧪 Evaluation

The system was evaluated on a dataset of X Japanese SMS messages, with Y% of them being scams. The best performing model achieved an accuracy of Z% and a false positive rate of just W%.

*Note: Replace X, Y, Z, and W with actual values from your dataset and experiments*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/Edge-AI-powered-Japanese-SMS-Scams-Detecting-Using-ONNX-Runtime](https://github.com/yourusername/Edge-AI-powered-Japanese-SMS-Scams-Detecting-Using-ONNX-Runtime)
