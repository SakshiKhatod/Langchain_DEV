"""
user gives text- need to have notes and quiz both-combine them and send it to user
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_Anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')


template1 = PromptTemplate(
    template="Generate me small and simple notes about following text \n{text}",
    input_variables=["text"],
)

template2 = PromptTemplate(
    template="Generate me 5 quiz questions based on the following text \n{text}",
    input_variables=["text"],
)

template3 = PromptTemplate(
    template="Combine the following notes and quiz questions into a single document \nNotes- {notes} ---->\nQuiz---->\n{quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain=RunnableParallel({
    "notes": template1 | model1 | parser,
    "quiz": template2 | model2 | parser,
})

merger_chain=template3 | model1 | parser #you can have with model 2 since you are merging only

final_chain=parallel_chain | merger_chain

text="""
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""
result = final_chain.invoke({"text": text})

print(result)

final_chain.get_graph().print_ascii() # this will print the graph of the chain
