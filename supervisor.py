import datetime
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from agents import (
    news,
    price,
    rag,
)
from langchain_core.messages import AIMessage
from langchain_core.messages import convert_to_messages


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

supervisor_prompt = f"""
You are Portfolio-GPT Supervisor.

Available specialists:
- rag      → purchase info (JSON)
- price    → live price by date (JSON)
- news     → headlines & sentiment

### Workflow rule for profit / loss
When the user asks for **profit, loss, P/L, gain, return**:

1. Call **rag** first to get purchase_price and shares.  
2. Then call **price** with today's date ({datetime.date.today()}) to get current price.  
3. Compute profit = (current – purchase_price) × shares, and percentage = profit / (purchase_price × shares) × 100.  
4. Reply to the user:  
   "<TICKER>: bought DATE at $X × N shares → current $Y → **+-Z% / +-$P**".

Call exactly one agent per step and always hand work back to yourself after each call.
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




for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What’s my AAPL position?",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]
for msg in reversed(final_message_history):
    if isinstance(msg, AIMessage):
        print("\n🧠 Final AI Message:\n")
        print(msg.content)
        break



