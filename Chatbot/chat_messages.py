from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)           
model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of India?"),
   
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)