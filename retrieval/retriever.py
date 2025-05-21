import binascii
from base64 import b64decode
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

def is_valid_base64(s: str) -> bool:
    try:
        return bool(s and b64decode(s, validate=True))
    except (binascii.Error, ValueError):
        return False

def parse_docs(docs):
    b64_images = []
    texts = []
    for doc in docs:
        content = getattr(doc, "page_content", str(doc))  
        if is_valid_base64(content):
            b64_images.append(content)
        else:
            if content.strip():
                texts.append(Document(page_content=content))
    return {"images": b64_images, "texts": texts}

def build_prompt(kwargs):
    context, question = kwargs["context"], kwargs["question"]
    prompt = [{"type": "text", "text": f"""Answer based only on the following context:\n{chr(10).join(d.page_content for d in context["texts"])}\n\nQuestion: {question}"""}]
    prompt += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in context["images"] if img.strip()]
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])

def create_rag_chain(retriever):
    return (
        {"context": retriever | RunnableLambda(parse_docs), "question": RunnablePassthrough()}
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
