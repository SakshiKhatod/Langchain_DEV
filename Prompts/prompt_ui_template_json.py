from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import streamlit as st

# load environment variables from .env file
load_dotenv()
# import models
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# write header and selectbox from streamlit
st.header("Research Assistant")
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)

template = load_prompt("prompt_template.json")

prompt = template.invoke(
    {
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
    }
)

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
