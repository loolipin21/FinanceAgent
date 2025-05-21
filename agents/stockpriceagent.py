import json
import yfinance as yf
import datetime
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


@tool(
    "get_stock_price",
    description="Fetches the closing stock price for a given company on a specific date using its ticker symbol. For example, 'AAPL' for Apple, 'MSFT' for Microsoft."
)
def get_stock_price(symbol: str, date: str) -> str:
    """
    This tool retrieves the historical closing price for a specified stock symbol on a given date.
    It uses Yahoo Finance's API to fetch stock data and returns the closing price formatted as a string.
    Args:
        symbol: the first argument
        date: the second argument
    """
    date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
    
    ticker = yf.Ticker(symbol)
    
    hist = ticker.history(start=date_obj, end=date_obj + datetime.timedelta(days=1))
    
    if len(hist) == 0:
        return f"Stock price data for {symbol} on {date} isn't available"
        
    price = hist['Close'][0]
    return json.dumps({
         "ticker": symbol,
         "date": date,
        "close": round(float(price), 2)
    })


@tool(
    "get_price_trend",
    description="Calculates the stock price trend over the past N days for a given ticker symbol. It computes the percentage change from the starting price to the most recent closing price."
)
def get_price_trend(ticker: str, days: int = 7) -> str:
    """
    Args:
        ticker: Stock symbol (e.g. 'GOOGL', 'NVDA')
        days: Number of most recent trading days to compute trend (default: 7)

    Returns:
        The percentage change and trend direction over the specified time range.
    """
    data = yf.Ticker(ticker).history(period=f"{days}d")

    if len(data) < 2:
        return f"Not enough data to calculate {days}-day trend for {ticker}."

    start = data["Close"].iloc[0]
    end = data["Close"].iloc[-1]
    pct_change = ((end - start) / start) * 100

    trend = "up" if pct_change > 0 else "down"
    return f"{ticker} is {trend} {abs(pct_change):.2f}% over the last {days} days."

TODAY = datetime.date.today().strftime("%Y-%m-%d")

financial_stock_prompt = """
You are a financial assistant agent using ReAct-style reasoning.
You help users retrieve accurate stock price information.

You have access to two tools:

1. `get_stock_price(ticker: str, date: str)`
   → Use this when the user asks for the price of a stock on a specific date, or uses words like “current”, “today”, or “now”.

2. `get_price_trend(ticker: str, days: int)`
   → Use this when the user asks about recent performance, trend, or movement over time (e.g., past 7 days, last week).

---

THINK → DECIDE → ACT

When you receive a user query:
- If the user gives a **company name** (e.g. “Apple”), convert it to its stock ticker (e.g. “AAPL”).
- If the user asks for:
  - “current price”, “today’s price”, or “price now” → use `get_stock_price(ticker, date=TODAY)`
  - A specific historical date → use `get_stock_price(ticker, date)`
  - Recent trend, performance, movement → use `get_price_trend`

---

EXAMPLES:

- "What was AAPL's price on May 10?" → use `get_stock_price("AAPL", "2024-05-10")`
- "What is MSFT's price today?" → use `get_stock_price("MSFT", TODAY)`
- "How has TSLA moved in the past week?" → use `get_price_trend("TSLA", 7)`

---

OUTPUT RULES:

- Always return the json output of the tool you used.
- Do NOT guess values or create commentary.
- If no data is found, return the tool’s fallback message.
"""



price = create_react_agent(model=ChatOpenAI(model="gpt-4o-mini"),
                                     tools=[get_stock_price, get_price_trend],
                                     name="price",
                                     prompt=financial_stock_prompt)

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

        for step in price.stream(
            {"messages": [system_msg, user_msg]},  
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main() 