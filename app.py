import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import json
import os

# Load configuration if available
def load_config(model_dir):
    config_path = os.path.join(model_dir, "absa_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_model():
    model_dir = "./final-absa-model"
    config = load_config(model_dir)
    
    # Default values based on your configuration
    base_model_name = config.get("model_name", "roberta-base") if config else "roberta-base"
    num_labels = config.get("num_labels", 3) if config else 3
    id2label = config.get("id2label", {"0": "positive", "1": "negative", "2": "neutral"}) if config else {"0": "positive", "1": "negative", "2": "neutral"}
    
    # Load tokenizer from the fine-tuned model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id={v: int(k) for k, v in id2label.items()}
    )
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    # Set to evaluation mode
    model.eval()
    
    return tokenizer, model, id2label

# --- Streamlit UI ---
st.set_page_config(page_title="Aspect-Based Sentiment Analysis", page_icon="💬")

st.title("Aspect-Based Sentiment Analysis")
st.write("Deploying the fine-tuned RoBERTa model with LoRA using Streamlit.")

# Load model and tokenizer
with st.spinner("Loading model... This might take a minute on first run."):
    tokenizer, model, id2label = load_model()

st.success("Model loaded successfully!")

# Input fields
user_sentence = st.text_area("Enter a review or sentence to analyze:", height=100, placeholder="Example: The battery life is amazing but the screen is a bit dim.")
user_aspect = st.text_input("Enter the aspect you want to analyze:", placeholder="Example: battery")

if st.button("Analyze Sentiment"):
    if user_sentence.strip() == "":
        st.warning("Please enter a sentence to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Aspect is now optional; if empty, it's passed as an empty string
            inputs = tokenizer(user_sentence, user_aspect, return_tensors="pt", truncation=True, max_length=96)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_id = torch.argmax(probabilities, dim=-1).item()
                
            # Get label and confidence
            predicted_label = id2label[str(predicted_class_id)]
            confidence = probabilities[0][predicted_class_id].item() * 100
            
            # Create JSON result
            result_json = {
                "text": user_sentence,
                "aspect": user_aspect if user_aspect.strip() != "" else "General",
                "prediction": predicted_label,
                "confidence": f"{confidence:.2f}%",
                "probabilities": {id2label[str(idx)]: float(prob) for idx, prob in enumerate(probabilities[0])}
            }
            
            # Save to a local file
            with open("results.json", "w") as f:
                json.dump(result_json, f, indent=4)

            # Display result
            st.markdown("### Result:")
            
            # Color mapping for different sentiments
            color = "gray"
            if predicted_label.lower() == "positive":
                color = "green"
            elif predicted_label.lower() == "negative":
                color = "red"
            elif predicted_label.lower() == "neutral":
                color = "blue"
                
            if user_aspect.strip() != "":
                st.markdown(f"**Aspect:** `{user_aspect}`")
            else:
                st.markdown(f"**Aspect:** `General / Not specified`")
                
            st.markdown(f"**Sentiment:** <span style='color:{color}; font-size:18px;'>{predicted_label.capitalize()}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            # Show all probabilities
            st.markdown("#### Confidence breakdown:")
            for idx, prob in enumerate(probabilities[0]):
                label = id2label[str(idx)]
                st.progress(float(prob), text=f"{label.capitalize()}: {prob.item()*100:.1f}%")

            # JSON Export Section
            st.divider()
            st.markdown("### Export Results")
            st.json(result_json)
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(result_json, indent=4),
                file_name="sentiment_results.json",
                mime="application/json"
            )
