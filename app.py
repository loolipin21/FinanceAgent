import os
import datetime
from pathlib import Path
import shutil
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from preprocessing.summarize_pdf import ingest
from agents import news, price, rag
from agents.portfolio_rag import init_rag
from langchain_core.messages import AIMessage, convert_to_messages

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

# Prevent Streamlit from watching torch internals
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = ".*torch.*"

# ── Streamlit Page Setup ───────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Agent", page_icon="💹")
st.title("💹 Portfolio-GPT")

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# ── PDF Upload ─────────────────────────────────────────────────────────────────
st.subheader("📄 Upload a PDF")
UPLOAD_DIR = "streamlit_upload"
INDEX_DIR = Path("faiss_index_folder")
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded = st.file_uploader("Drop a PDF", type=["pdf"])
if uploaded:
    file_path = os.path.join(UPLOAD_DIR, uploaded.name)
    if file_path not in st.session_state.ingested_files:
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved to `{file_path}`")
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        ingest(file_path)
        init_rag("summaries.json")
        st.session_state.ingested_files.add(file_path)
    else:
        st.info(f"🔁 File already ingested: {uploaded.name}")

st.divider()

# ── Supervisor Agent ───────────────────────────────────────────────────────────
supervisor_prompt = f"""
You are Portfolio-GPT Supervisor — a high-level orchestrator responsible for answering investment-related questions by routing tasks to 3 specialized agents:

- **rag** → Portfolio data & investment analysis (from uploaded PDFs)
- **price** → Live stock prices and trends (via Yahoo Finance)
- **news** → Latest headlines & sentiment

---

THINK → DECIDE → ACT

Use the following routing rules to determine which agent to call:

---

1. PORTFOLIO ANALYSIS & COMMENTARY (rag only)
Trigger the **rag** agent when the user asks:
- “What’s my analysis?”
- “Give me a summary of my portfolio”
- “What do I think about my stocks?”
- “Show commentary”

Action:
→ Call `rag` directly and return its output.
Do NOT call price or news.

---

2. PROFIT / LOSS CALCULATION (rag + price)
If the user mentions:
- “profit”, “gain”, “loss”, “P/L”, “return on investment”

Step-by-step:
  a. Call **rag** to retrieve `purchase_price`, `shares`, and `purchase_date`
  b. Call **price** with today's date ({datetime.date.today()}) to get `current_price`
  c. Calculate:
     - `profit = (current_price - purchase_price) × shares`
     - `percentage = profit / (purchase_price × shares) × 100`
  d. Reply:
     "**<TICKER>**: bought on <DATE> at $X × N shares → current $Y → **±Z% / ±$P**" with the calculations explaining how the value is found
 Do NOT guess missing numbers. Skip tickers if data is incomplete.
 

3. FULL STOCK CHECK (rag + price + news)

If the user asks for a full update or opinion on a stock — e.g.,

- “What’s going on with my Apple shares?”
- “Give me an update on Tesla”
- “Tell me how my Microsoft holding is doing right now”
- “Any news and performance for my stocks?”

Step-by-step:
  a. Call **rag** to get user-owned tickers, number of shares, purchase price/date, and commentary.
  b. Call **price** to retrieve current stock price and performance trend.
  c. Call **news** to fetch current sentiment or headlines.

Combine all 3 to give a complete overview:

→ Example format:
> "**AAPL**: You bought 20 shares at $145.30 on May 10, 2023. Current price is $187.50 (↑29.00%). News sentiment: Positive — headlines suggest strong iPhone 16 demand and AI growth."

---

GLOBAL RULES:

- One agent per step. After every call, always hand control back to yourself.
- Never synthesize financial advice without retrieved evidence.
- Never guess. If a value is missing, skip or return partial analysis.
- Do not call more than needed — route precisely.

If the user’s question does not match any rule above, return:
- “I’m not sure which agent to route this to. Please clarify your question.”

"""


supervisor = (
    create_supervisor(
        model=ChatOpenAI(model="gpt-4o-mini"),
        agents=[news, price, rag],
        prompt=supervisor_prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",
    )
    .compile(name="portfolio_supervisor")
)

# Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input
prompt = st.chat_input("Ask something about your portfolio…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ph = st.empty()
        answer = ""

        with st.spinner("Thinking..."):
            today = datetime.date.today().strftime("%Y-%m-%d")
            system_msg = {
                "role": "system",
                "content": (
                    f"Today's date is {today}. "
                    "If the user says 'today', 'now', or 'current', interpret it as this date."
                ),
            }
            for chunk in supervisor.stream({"messages": [system_msg, {"role": "user", "content": prompt}]}):
                pretty_print_messages(chunk, last_message=True)

            final_message_history = chunk["supervisor"]["messages"]
            for msg in reversed(final_message_history):
                if isinstance(msg, AIMessage):
                    answer = msg.content
                    break
        
        answer.replace("$", r"\$")
        ph.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
