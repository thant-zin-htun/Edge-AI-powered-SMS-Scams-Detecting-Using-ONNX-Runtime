import streamlit as st
import onnxruntime as rt
import pickle
import numpy as np
import MeCab
import time

def analyzer(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip().split()

# Load the pre-trained CountVectorizer with the vocabulary from training.
with open('vectorizer.pkl', 'rb') as f: 
    vectorizer = pickle.load(f)

providers = [
    ('CoreMLExecutionProvider', {
        "ModelFormat": "MLProgram", 
        "MLComputeUnits": "ALL", 
        "RequireStaticInputShapes": "0", 
        "EnableOnSubgraphs": "0"
    }),
]

# Define the models and their paths
models = {
    "Logistic Regression": 'New_models/logistic_regression.onnx',
    "Random Forest": 'New_models/random_forest.onnx',
    "SVM": 'New_models/SVM.onnx'
}

# Add a language selection dropdown
language = st.selectbox("Choose a language / 言語を選択してください / ဘာသာစကားရွေးချယ်ပါ", ["English", "日本語", "မြန်မာ"])

# Define text content based on the selected language
if language == "English":
    title = "SMS Scam Shield: AI-Powered Fraud Detection"
    subtitle = "Protecting Your Privacy with Edge AI – Secure and Reliable Scam Detection on Your Device"
    model_selection_text = "Choose a model"
    message_input_text = "Enter the message"
    button_text = "Enter"
    prediction_text = "Prediction"
    latency_text = "Latency"
    warning_text = "Please enter a message."
elif language == "日本語":
    title = "SMS詐欺シールド：AI搭載の詐欺検出"
    subtitle = "エッジAIでプライバシーを保護 – デバイス上で安全で信頼性の高い詐欺検出"
    model_selection_text = "モデルを選択してください"
    message_input_text = "メッセージを入力してください"
    button_text = "入力"
    prediction_text = "予測"
    latency_text = "レイテンシ"
    warning_text = "メッセージを入力してください。"
else:
    title = "SMS Scam Shield: AI - အသုံးပြုထားသော လိမ်လည်မှုဖော်ထုတ်ရေး"
    subtitle = "Edge AI ဖြင့် သင့်ကိုယ်ရေးကိုယ်တာကို ကာကွယ်ပါ – သင့်စက်ပစ္စည်းပေါ်တွင် လုံခြုံစိတ်ချရသော လိမ်လည်မှုဖော်ထုတ်ရေး"
    model_selection_text = "မော်ဒယ်ကို ရွေးချယ်ပါ"
    message_input_text = "မက်ဆေ့ကို ရိုက်ထည့်ပါ"
    button_text = "ရိုက်ထည့်ပါ"
    prediction_text = "ခန့်မှန်းချက်"
    latency_text = "ကြာချိန်"
    warning_text = "မက်ဆေ့ကို ရိုက်ထည့်ပါ။"

# Add a header and subheader
st.title(title)
st.subheader(subtitle)

# Add a model selection dropdown
model_choice = st.selectbox(model_selection_text, list(models.keys()))

# Add a text input for the message
msg = st.text_input(message_input_text)

if st.button(button_text):
    if msg:
        # Load the selected model
        model_path = models[model_choice]
        ort_sess = rt.InferenceSession(model_path, providers=providers)
        input_name = ort_sess.get_inputs()[0].name
        
        # Use transform() to ensure the vocabulary remains the same as training.
        vec_msg = vectorizer.transform([msg])
        
        # Convert the sparse matrix to a dense NumPy array.
        vec_msg_dense = vec_msg.toarray().astype(np.float32)
        
        # Run the inference.
        start_time = time.time()
        pred_onx = ort_sess.run(None, {input_name: vec_msg_dense})[0]
        latency = time.time() - start_time
        
        # Display the result as "spam" or "ham"
        result = "spam" if pred_onx[0] == 1 else "ham"
        st.success(f"{prediction_text}: {result}")
        st.write(f"{latency_text}: {latency:.4f} seconds")
    else:
        st.warning(warning_text)

# Add some custom styling
st.markdown(
    """
    <style>
    .reportview-container {
        padding: 2rem;
        font-family: 'Noto Sans JP', sans-serif;
    }
    .main .block-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        max-width: 600px; /* Shorten the block length */
        margin: auto;
    }
    .stButton>button {
        background-color: #E60012;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        font-family: 'Noto Sans JP', sans-serif;
    }
    .stTextInput>div>div>input {
        border: 2px solid #E60012;
        border-radius: 4px;
        padding: 10px;
        font-family: 'Noto Sans JP', sans-serif;
    }
    .css-1lcbmhc {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)