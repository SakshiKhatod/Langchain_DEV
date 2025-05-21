from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """ My name is Sakshi
I am 27 years old

I live in Gurgaon 
How are you?
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    # length_function=len,
    # separators=["\n\n", "\n", " ", ""],
    # keep_separator=True,
    # add_start_index=True,
    # add_end_index=True,
)

result = splitter.split_text(text)
print(result)
print(len(result))
