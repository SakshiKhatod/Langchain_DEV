"""
User gives a feedback on any product.
Analyzing the feedback we hve to fuind out the sentiment-positive fdbk or negative fdbk
"""

# from lanchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from pydantic import BaseModel, Field
# from typing import Literal


# load_dotenv()
# model=ChatOpenAI()
# parser=StrOutputParser()


# prompt1=PromptTemplate(
#     template="Analyze the feedback and classify the sentiment whether positive or negatve of the feedback. \nFeedback: {feedback}",
#     input_variables=["feedback"]
# )

# chain=prompt1 | model | parser

# print(chain.invoke({'feedback': 'This is a beautiful phone'})) # positive feedback
# #here the output should be consistent- like everytime you should get positive or negative feedbackas nt Positive/positive/pos.
# # as you are giving to llm only so the output should be consistent and not any other hence we wil use pydantic class




from lanchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch,RunnableLambda

load_dotenv()
model=ChatOpenAI()
parser=StrOutputParser()

# pydantic class for the structured output
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"]=Field(description="Give me Sentiment of the feedback")


parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="Analyze the feedback and classify the sentiment whether positive or negatve of the feedback. \nFeedback: {feedback} \n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions() } 
)

# result=chain.invoke({'feedback': 'This is a beautiful phone'}).sentiment  #sentiment='positive' always now
# print(result) # positive


prompt2=PromptTemplate(
    template="Write an appropriate response for positive feedback",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="Write an appropriate response for negative feedback",
    input_variables=["feedback"]
)


classifier_chain=prompt1 | model | parser2

branch_chain=RunnableBranch(
   (lambda x: x.sentiment=='positive',prompt2|model|parser), 
   (lambda x: x.sentiment=='negative',prompt3|model|parser), 
   RunnableLambda(lambda x:"could not find sentiment")
)

final_chain=classifier_chain|branch_chain
result=final_chain.invoke({'feedback': 'This is a beautiful phone'})
print(result) # positive feedback

final_chain.get_graph().print_ascii()

