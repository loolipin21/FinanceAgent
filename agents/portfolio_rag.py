#mandatory imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from retrieval.faiss_store import build_retriever
from retrieval.retriever import create_rag_chain
from preprocessing.summarize_pdf import ingest
from langchain_openai import ChatOpenAI

# Optional: if running for first time
# ingest("tests/test2.pdf")  # or dynamically load PDF
# retriever = build_retriever("summaries.json")
# rag_chain = create_rag_chain(retriever)

retriever = None       
rag_chain = None

def init_rag(summary_path: str):
    """Build (or rebuild) the retriever + chain."""
    global retriever, rag_chain
    retriever = build_retriever(summary_path)
    rag_chain = create_rag_chain(retriever)

@tool("answer_investment_question")
def answer_investment_question(question: str) -> str:
    """Answers investment-related questions using previously summarized PDFs.
    """

    if retriever is None:
        return "No documents ingested yet. Please upload a PDF."

    docs = retriever.invoke(question)
    for i, doc in enumerate(docs):
        if hasattr(doc, "page_content"):
            print(f"\nRetrieved context {i + 1}:\n{doc.page_content}")
        else:
            print(f"\nRetrieved raw item {i + 1}:\n{doc}")

    # Now feed it into the chain
    return rag_chain.invoke(question)


rag_prompt = """
You are a financial portfolio analysis assistant using ReAct-style reasoning.
You answer questions based solely on extracted summaries from uploaded PDF reports (e.g., investment tables and commentary sections).

---
 TOOL: `answer_investment_question(question: str)`
Use this tool when the user asks about:

- Portfolio composition
- Stock purchase history (date, price, quantity)
- PDF-based commentary or analysis
- Past performance summaries inside the document
- Reasoning behind stock purchases
- General portfolio summary or “What do you think of my stocks?”

---

 DO NOT handle questions like:
- Current stock prices → handled by the **price** agent.
- Recent market news or sentiment → handled by the **news** agent.
- Future predictions, advice, or financial strategy.

---

 BEHAVIOR RULES:

- ONLY use information retrieved from the PDF.
- DO NOT invent or guess answers.
- DO NOT infer user goals, strategy, or emotion.
- DO NOT call other agents or APIs.
- DO NOT add your own commentary or opinion.

---

 EXAMPLES:

User: "What’s my analysis on TSLA?"
→ Use `answer_investment_question("What’s my analysis on TSLA?")`

User: "When did I buy Apple?"
→ Use `answer_investment_question("When did I buy Apple?")`

User: "How has my portfolio performed?"
→ Use `answer_investment_question("How has my portfolio performed?")`

User: "What’s AAPL’s price today?"
→ DO NOT answer — this belongs to the **price** agent.

---
OUTPUT RULES (MANDATORY):
- If data is found → reply **with nothing except** a valid JSON object:
  ```
  {
    "ticker": "<TICKER>",
    "purchase_price": <float>,
    "purchase_date": "<YYYY-MM-DD>",
    "shares": <int>
  }
  ```
- If an element is unknown, omit it (don’t invent it).
- If no answer → reply exactly `"NOT_FOUND"`.
"""



rag = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[answer_investment_question],
    name="rag",
    prompt=rag_prompt
)

def main():
    while True:
        query = input("Ask a portfolio question (or 'exit'): ")
        if query.lower() in {"exit", "quit"}:
            break

        for step in rag.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
