from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser,ResponseSchema #not in core

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

schema=[ ResponseSchema(name="fact_1",description="fact 1 about the topic"),
        ResponseSchema(name="fact_2",description="fact 2 about the topic"),
        ResponseSchema(name="fact_3",description="fact 3 about the topic")]


parser=StructuredOutputParser.from_response_schemas(schema)
template=PromptTemplate(
    template="Write 3 facts about o {topic} \n {format_instruction}", input_variables=["topic"],
    partial_variables={'format_instruction':parser.get_format_instructions() } #returns json object
)   

#without chain
prompt=template.invoke({"topic":"black hole"})
result = model.invoke(prompt)


#this will be final result in json format
final_result=parser.parse(result.content)
print(result)
print(final_result) # final_result is a dictionary with keys as fact_1, fact_2, fact_3 and values as the facts about black hole.

#with chain
chain=template | model | parser
result=chain.invoke({"topic":"black hole"})
print(result) # final_result is a dictionary with keys as fact_1, fact_2, fact_3 and values as the facts about black hole.