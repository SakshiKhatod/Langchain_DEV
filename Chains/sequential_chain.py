from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)

model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)


template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n{text}",
    input_variables=["text"],
)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)

# prompt=template1.invoke({"topic": "black hole"})

# result = model.invoke(prompt)

# prompt1=template2.invoke({"text": result.content})

# result1 = model.invoke(prompt1)
# print(result1.content)
# # print(result1) # this will be a string with 5 line summary of the detailed report on black hole.
