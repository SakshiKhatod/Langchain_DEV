from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
# load chat history
# create prompt

# chat template
chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("Human", "{query}"),
    ]
)
# load chat history
chat_history = []
with open("chat_history.txt") as f:
    chat_history.extend(f.readLines())

print(chat_history)
# create prompt

prompt = chat_history.invoke(
    {"chat_history": chat_history, "query": "Where is my refund?"}
)

# send to llm and run the model
