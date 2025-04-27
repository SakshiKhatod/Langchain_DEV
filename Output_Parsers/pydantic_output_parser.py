from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Load environment variables from .env file
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person belonging to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me name,age and city of fictional {place} person\n {format_instruction}",
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()  # returns pydntic object
    },
)

prompt = model.invoke({"place": "Indian"})
result = model.invoke(prompt)
print(result)

final_result = parser.parse(result.content)
print(
    final_result
)  # final_result will return text with keys as name, age, city name="Ramesh Sharma", age=25, city="New York"
