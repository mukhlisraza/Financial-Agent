from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv
openai.api_key=os.getenv("OPENAI_API_KEY")

# Web search agent
web_search_agent=Agent(
    name="Web search agent",
    role="search the web for information",
    model=Groq(id="llama3-8b-8192"),
    tools=[DuckDuckGo()], 
    instructions=["Always provide me sources"],
    show_tools_calls=True,
    markdown=True,   
)

# Financial Agent
finance_agent=Agent(
    name="Finance AI agent",
    model=Groq(id="llama3-8b-8192"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tools_calls=True,
    markdown=True,   
)

multi_ai_agent = Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,   
)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NV", stream=True)