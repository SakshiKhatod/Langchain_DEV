from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

os.environ["HF_HOME"] = (
    "D:/huggingface_cache"  # Set the cache directory for Hugging Face models
)

# load_dotenv()


llm = HuggingFacePipeline.from_model_id(
    model_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 100},
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
print(result.content)
