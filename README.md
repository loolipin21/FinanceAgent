# 💹 Portfolio-GPT: AI Investment Agent with LangGraph & Streamlit

Portfolio-GPT is an **agentic AI system** for analyzing your stock portfolio using uploaded PDF reports. It leverages **LangGraph**, **Streamlit**, and **multi-agent reasoning** to answer investment questions with grounded evidence and real-time insights.

---

## 🚀 Features

**PDF Portfolio Ingestion**  
Upload your investment reports — Portfolio-GPT extracts structured tables, charts, and narratives using advanced parsing.

**Multi-Agent Reasoning (LangGraph Supervisor)**  
Your queries are routed through 3 intelligent agents:

- `rag` → Retrieves your portfolio purchases & commentary from PDF  
- `price` → Fetches live stock prices & trends via Yahoo Finance  
- `news` → Analyzes recent headlines and sentiment per ticker  

✅ **End-to-End Investment Queries**  
Ask complex financial questions like:

- “How much profit did I make on AAPL?”  
- “Whats the sentiment on AAPL”  

✅ **Intelligent Prompt Engineering**  
Prompts use ReAct-style reasoning and enforce agent-by-agent routing to prevent hallucination and ensure tool usage.

---

## 🧠 Example Workflow

Ask:
> “Should I sell my Microsoft shares?”

**Portfolio-GPT will:**
1. Use `rag` to retrieve your purchase price, date, and analysis  
2. Use `price` to get the current stock price and trend  
3. Use `news` to analyze sentiment  
4. Return a full response like:

> “You bought MSFT on 2022-11-15 at $242.50. It is now $458.17 (+89.0%) with mixed sentiment. Holding may be wise.”

## 📦 Requirements

- Python 3.10+
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Streamlit](https://streamlit.io/)
- `yfinance`, `unstructured`, `pymupdf4llm`, `transformers`, `openai`

Install:
```bash
pip install -r requirements.txt

```

Running The App
```bash
streamlit run app.py
```
