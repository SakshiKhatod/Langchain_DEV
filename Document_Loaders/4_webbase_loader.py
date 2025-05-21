from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# url = "https://www.allaboutvlsi.com/apb-protocol"
url = "https://github.com/campusx-official/langchain-document-loaders/blob/main/dl-curriculum.pdf"
loader = WebBaseLoader(url)
data = loader.load()
# print(data[0].page_content)
# print(data[0].metadata)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"],
)

parser = StrOutputParser()


chain = prompt | model | parser

# print(
#     chain.invoke(
#         {"question": "What is full form of APB?", "text": data[0].page_content}
#     )
# )


# print(
#     chain.invoke(
#         {
#             "question": "What are the four states of state machine?",
#             "text": data[0].page_content,
#         }
#     )
# )


print(
    chain.invoke(
        {
            "question": "What is the summary that we are talking about?",
            "text": data[0].page_content,
        }
    )
)
