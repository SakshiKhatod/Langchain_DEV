from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# load environment variables from .env file
load_dotenv()
# import models
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)
model = ChatHuggingFace(llm=llm)
chat_history = []
while True:
    user_input = input("User: ")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ", result.content)

print("Chat History:", chat_history)
