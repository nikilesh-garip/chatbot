import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("🤖 Chat with Hugging Face Model")

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

chat_input = st.text_input("You:", "")

if chat_input:
    inputs = tokenizer(chat_input + tokenizer.eos_token, return_tensors="pt")
    with st.spinner("Thinking..."):
        reply_ids = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    st.write("🤖:", reply)
