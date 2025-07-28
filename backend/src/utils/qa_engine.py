from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()

    prompt_template = PromptTemplate.from_template("""
You are a financial analyst. Use the following context from a company's annual report to answer the question.
If the answer isn't in the context, say "Not found in the document."

Context:
{context}

Question: {question}
Answer:
""")

    llm = ChatGroq(model="llama3-8b-8192")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
