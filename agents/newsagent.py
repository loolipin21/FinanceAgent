import yfinance as yf
import datetime
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from transformers import pipeline


@tool("get_finance_news")
def get_finance_news(query: str) -> str:
    """
    Fetches the latest financial news headlines related to a stock ticker (e.g. 'AAPL' or 'TSLA').
    Args:
        query (str): A stock ticker symbol (e.g., "MSFT", "NVDA").
    
    Returns:
        str: A newline-separated list of recent news headlines for the specified ticker. 
             If no news is found, a message indicating that is returned.
    """
    yfnewstool = YahooFinanceNewsTool()

    return yfnewstool.invoke(query)

sentiment_pipeline = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
# sentiment_pipeline = pipeline("sentiment-analysis")

@tool("summarize_news_tone")
def summarize_news_tone(ticker: str) -> str:
    """
    Fetches recent news headlines for a given stock ticker, analyzes their sentiment,
    and returns a summary of the overall tone.

    Args:
        ticker (str): A stock ticker symbol (e.g., "GOOGL", "AMZN").
    
    Returns:
        str: A summary of the overall sentiment (e.g., "POSITIVE") based on recent headlines.
             Includes a count breakdown (positive, negative, neutral) and 2–3 example headlines with sentiment labels.
             If no headlines are found, a fallback message is returned.
    """
    raw_headlines = YahooFinanceNewsTool().run(ticker)
    headlines = [h.strip() for h in raw_headlines.split("\n") if h.strip()]

    if not headlines:
        return f"No recent news headlines found for {ticker}."

    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    scored_headlines = []

    for h in headlines[:5]:  
        result = sentiment_pipeline(h)[0]
        label = result["label"].upper()
        score = result["score"]

        if label == "POSITIVE" and score < 0.7:
            label = "NEUTRAL"
        elif label == "NEGATIVE" and score < 0.7:
            label = "NEUTRAL"

        sentiment_counts[label] += 1
        scored_headlines.append((h, label, score))

    majority = max(sentiment_counts, key=sentiment_counts.get)

    summary = f"Overall sentiment for {ticker}: **{majority}**\n"
    breakdown = (
        f"POSITIVE: {sentiment_counts['POSITIVE']}, "
        f"NEGATIVE: {sentiment_counts['NEGATIVE']}, "
        f"NEUTRAL: {sentiment_counts['NEUTRAL']}"
    )
    examples = "\n".join(
        f"- {label} → \"{text}\"" for text, label, _ in scored_headlines[:3]
    )

    return f"{summary}{breakdown}\n\nHeadlines:\n{examples}"

TODAY = datetime.date.today().strftime("%Y-%m-%d")

news_sentiment_prompt = """
You are a financial news sentiment agent using ReAct-style reasoning.

You help users:
1. Fetch the latest financial news for public companies.
2. Summarize the overall tone (sentiment) of recent headlines.

---

TOOLS AVAILABLE:

- `get_finance_news(ticker: str)`
  → Use this when the user asks to “see news”, “show recent headlines”, or “find updates” about a company.

- `summarize_news_tone(ticker: str)`
  → Use this when the user asks about the **sentiment**, **tone**, **mood**, or “what do people feel” about a stock.

---

THINK → DECIDE → ACT

- If the user gives a **company name** (e.g. “Apple”), convert it to its stock ticker (e.g. “AAPL”).
- If the query is about “news”, “latest articles”, or “what’s happening”, use `get_finance_news`.
- If the query is about “sentiment”, “tone”, “positive or negative”, or “market perception”, use `summarize_news_tone`.

Never guess or invent sentiment. Always use the tools.

---

EXAMPLES:

User: “What are people saying about Microsoft?”
→ Convert to MSFT → use `summarize_news_tone("MSFT")`

User: “Show me the latest news on Tesla”
→ Convert to TSLA → use `get_finance_news("TSLA")`

User: “What’s the market tone for AAPL?”
→ Use `summarize_news_tone("AAPL")`

---

OUTPUT RULES:

- Always return the tool result exactly as received.
- If no headlines are found, respond: “No recent news headlines found for <ticker>.”
- Do NOT generate your own summaries or tone. Use only what the tool provides.
- Format your final answer like this:

**Response:**
<result>

"""



news = create_react_agent(model=ChatOpenAI(model="gpt-4o-mini"),
                                     tools=[get_finance_news, summarize_news_tone],
                                     name="news",
                                     prompt=news_sentiment_prompt)

def main():
    today = datetime.date.today().strftime("%Y-%m-%d")
    system_msg = {
        "role": "system",
        "content": (
            f"Today's date is {today}. "
            "If the user says 'today', 'now', or 'current', interpret it as this date."
        ),
    }

    while True:
        query = input("→ ")
        if query.lower() in {"exit", "quit"}:
            break

        user_msg = {"role": "user", "content": query}

        for step in news.stream(
            {"messages": [system_msg, user_msg]},  
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main() 