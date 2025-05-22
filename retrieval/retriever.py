from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

def parse_docs(docs):
    """
    Keep only valid text documents (filters out empty/malformed ones).
    Returns format compatible with original multimodal prompt structure.
    """
    texts = []
    for doc in docs:
        content = getattr(doc, "page_content", str(doc))
        if content.strip():
            texts.append(Document(page_content=content))
    return { "texts": texts}  

def build_prompt(kwargs):
    """
    Builds a prompt using only clean text docs.
    Preserves multimodal message structure for future compatibility.
    """
    context, question = kwargs["context"], kwargs["question"]
    prompt = [{
        "type": "text",
        "text": f"""Answer based only on the following context:\n{chr(10).join(d.page_content for d in context["texts"])}\n\nQuestion: {question}"""
    }]
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])

def create_rag_chain(retriever):
    """
    Final RAG chain.
    """
    return (
        {"context": retriever | RunnableLambda(parse_docs), "question": RunnablePassthrough()}
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
