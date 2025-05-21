# from langchain_community.document_loaders import TextLoader

# loader = TextLoader("sample.txt", encoding="utf-8")
# data = loader.load()
# print(data)
# print(type(data))
# print(type(data[0]))
# print(data[0].page_content)
# print(data[0].metadata)


# import os
# from langchain_community.document_loaders import TextLoader

# current_dir = os.path.dirname(__file__)
# file_path = os.path.join(current_dir, "sample.txt")

# loader = TextLoader(file_path, encoding="utf-8")
# data = loader.load()
# print(data)


from langchain_community.document_loaders import TextLoader

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

# model = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
# )

model = ChatHuggingFace(llm=llm)

loader = TextLoader("sample.txt", encoding="utf-8")
data = loader.load()

prompt = PromptTemplate(
    template="Write a detailed summary on {poem}", input_variables=["poem"]
)
parser = StrOutputParser()
chain = prompt | model | parser

output = chain.invoke({"poem": data[0].page_content})
print(output)


"""

The poem celebrates the simplicity of the game, where young boys chase after runs barefoot in the lanes and the legends 
of cricket rise from humble beginnings. It portrays the excitement and energy of packed arenas where the game unites a global crowd.                                                                                                                                                                                                                                    The poet observes the tense moments of a cricket match, from the captains' choice of bat or ball to the ball's flight through the air. He illustrates the game's intricacies and unpredictability, where every run and wicket is important, from the flair of the openers to the courageous tail-enders.                                                                                                                                                                                        Moreover, the poem highlights the use of technology in the sport and the role it plays in deciding the outcome of the game. The tension and passion that permeates the game is audible through the crowd's roars and cheers.                                                                                                                                            The poem captures the emotional connection between cricket and its fans, who are united by a shared experience of heartbreak, thrill, and something more.                                                                                                                                                                                                               The author portrays cricket as an enduring sport that goes beyond mere sport, representing blood and lore, connecting families and generations across cultural and linguistic divides.                                                                                                                                                                                  The poem concludes with a call to celebrate cricket's magic and the love and fond memories 
it continues to evoke. Cricket, as the author underlines, is a gift, a flame, and a timeless thread that connects us all. 
"""