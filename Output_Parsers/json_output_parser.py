from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser


# Load environment variables from .env file
load_dotenv()
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="Give me name,age and city of fictional person\n {format_instruction}",
    input_variables=["format_instruction"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()  # returns json object
    },
)
prompt = template1.format()#can be used as an alternative to invoke

result = model.invoke(prompt) 
print(result)
