"""
agent_setup.py
Builds the LangChain agent: picks the LLM, attaches the tools and runs queries.
"""
import os
import time
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from finance_tools import TOOLS
from offline_model import OfflineToolRouter

SYSTEM_PROMPT = """You are a personal finance assistant for users in India. Amounts are in rupees (₹).
Always use the provided tools for any calculation. Never do the arithmetic yourself.
Convert lakh and crore to plain numbers before calling a tool (1 lakh = 100000, 1 crore = 10000000).
Pick the tool that matches the question:
- loans and EMI -> emi_calculator
- fixed monthly investment and final value -> sip_calculator
- target amount and monthly saving needed -> goal_sip_calculator
- budget for an income -> budget_planner
- anything else numeric (percentages, totals) -> calculator
If the user leaves out a number the tool needs (for example the return rate), say which assumption you used.
Answer in a short, plain way and quote the numbers returned by the tool.
If the question is not about money or calculations, say that you can only help with personal finance."""


def get_llm():
    """Gemini if a Google key is set, GPT-4o-mini if an OpenAI key is set, else the offline model."""
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        llm = ChatGoogleGenerativeAI(model=name, google_api_key=google_key,
                                     temperature=0, max_retries=1)
        return llm, f"{name} (Google)"
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0), "gpt-4o-mini (OpenAI)"
    return OfflineToolRouter(), "offline test model (no API key found)"


def build_agent():
    llm, label = get_llm()
    agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    return agent, label


def reply_text(content):
    """Gemini returns a list of content blocks (text plus a signature); join the text parts."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def run_query(agent, question, tries=4, wait=20):
    """Run one question; return the tool calls that were made and the final answer.
    Google answers 503 when it is overloaded, so those are retried a few times."""
    for attempt in range(tries):
        try:
            state = agent.invoke({"messages": [{"role": "user", "content": question}]})
            break
        except Exception as err:
            busy = "503" in str(err) or "UNAVAILABLE" in str(err)
            if busy and attempt < tries - 1:
                time.sleep(wait)
                continue
            raise
    calls = []
    for m in state["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            calls += [(c["name"], c["args"]) for c in m.tool_calls]
    answer = reply_text(state["messages"][-1].content)
    return {"tools_used": calls, "answer": answer, "messages": state["messages"]}
