from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage no need for ChatPromptTemplate


# chat_template=ChatPromptTemplate([
#     SystemMessage(content="You are a {domain} expert."),
#     HumanMessage(content="Explain in simple terms: {topic}")

# ])

# prompt=chat_template.invoke({
#     'domain':"cricket",
#     'topic':"free hit"}
# )  doesn't work same as Prompt Template

# need to send a s a tuple
chat_template = ChatPromptTemplate(
    [
        ("system", "You are a {domain} expert."),
        ("Human", "Explain in simple terms: {topic}"),
    ]
)

prompt = chat_template.invoke({"domain": "cricket", "topic": "free hit"})
print(prompt)  # this is the prompt that will be sent to the model
