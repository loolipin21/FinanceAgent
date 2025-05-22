import json
import uuid
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Directory to store the FAISS index and optional document store
INDEX_DIR   = "faiss_index_folder"                 
DOCSTORE_P  = Path(INDEX_DIR) / "docstore.pkl"     

def build_retriever(summary_path: str = "summaries.json") -> MultiVectorRetriever:
    """
    Returns a ready-to-use MultiVectorRetriever.
    Loads an existing FAISS index if available; otherwise builds a new one from the provided JSON summary file.
    """

    # Load previously saved vector index and document store (if available)
    if Path(INDEX_DIR).is_dir() and DOCSTORE_P.exists():
        print("🔄  Loading existing FAISS index & doc-store …")
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True 
        )
        with DOCSTORE_P.open("rb") as f:
            docstore = pickle.load(f)

        return MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id",
        )

    # Build new vector index from summaries
    print("🛠️   Building FAISS index from", summary_path)
    with open(summary_path, "r") as f:
        all_data = json.load(f)

    # Separate each type of content by source
    texts, tables, charts = [], [], []
    text_sum, table_sum, chart_sum = [], [], []

    for item in all_data:
        if item["type"] == "text":
            # Narrative PDF content 
            raw = item["raw"]
            texts.append(raw)
            text_sum.append(raw)

        elif item["type"] == "table":
            # Parsed table summary from markdown or PDF tables
            tables.append(item["raw"])
            table_sum.append(item["summary"])

        elif item["type"] == "chart":
            # LLM-interpreted chart data (from image-to-JSON)
            charts.append(item["extracted"])
            chart_sum.append(json.dumps(item["extracted"]))

    # Assign UUIDs so we can trace each document if needed (not used)
    text_ids  = [str(uuid.uuid4()) for _ in text_sum]
    table_ids = [str(uuid.uuid4()) for _ in table_sum]
    chart_ids = [str(uuid.uuid4()) for _ in chart_sum]

    # Create documents to embed into the FAISS vector store
    summary_docs = (
        [Document(page_content=s, metadata={"doc_id": i, "original": t})
         for s, i, t in zip(text_sum,  text_ids,  texts)]  +
        [Document(page_content=s, metadata={"doc_id": i, "original": t})
         for s, i, t in zip(table_sum, table_ids, tables)] +
        [Document(page_content=s, metadata={"doc_id": i, "original": t})
         for s, i, t in zip(chart_sum, chart_ids, charts)]
    )

    # Build and save the FAISS index
    vectorstore = FAISS.from_documents(summary_docs, OpenAIEmbeddings())
    vectorstore.save_local(INDEX_DIR)

    #Build and save docstore (optional — used for full-text traceability)
    docstore = InMemoryStore()
    docstore.mset(
        list(zip(text_ids,  text_sum)) +
        list(zip(table_ids, table_sum)) +
        list(zip(chart_ids, chart_sum))
    )
    DOCSTORE_P.parent.mkdir(parents=True, exist_ok=True)
    with DOCSTORE_P.open("wb") as f:
        pickle.dump(docstore, f)

    print("✅  Vectorstore & retriever are ready.")
    #Create Retriever
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
    )
