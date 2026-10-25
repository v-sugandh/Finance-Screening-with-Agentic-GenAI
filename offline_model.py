"""
offline_model.py
A tiny stand-in chat model that follows the same tool-calling protocol as a real LLM.

It is NOT a language model. It reads the question with regular expressions, asks for one
of our tools, and then repeats the tool output as the final reply. It exists so the agent
loop, the tools and the Streamlit app can be tested on a machine with no GoogleAI key.
agent_setup.py only falls back to it when GOOGLE_API_KEY is missing.
"""
import re
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

UNITS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7, "k": 1e3}


def _amounts(text):
    """Money values in order of appearance. Plain small numbers (years, rates) are skipped."""
    found = []
    pattern = r"(₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lac|crores?|cr|k)?\b"
    for m in re.finditer(pattern, text, re.I):
        value = float(m.group(2).replace(",", ""))
        unit = (m.group(3) or "").lower()
        if unit:
            value *= UNITS[unit]
        if m.group(1) or unit or value >= 1000:
            found.append(value)
    return found


def _rate(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", text, re.I)
    return float(m.group(1)) if m else None


def _years(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*months?", text, re.I)
    return float(m.group(1)) / 12 if m else None


def choose_tool(query):
    """Return (tool_name, args) or None if no tool fits."""
    q = query.lower()
    money, rate, years = _amounts(query), _rate(query), _years(query)

    pct_of = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of", q)
    if pct_of and money:
        return "calculator", {"expression": f"{pct_of.group(1)} / 100 * {money[0]:g}"}
    if ("emi" in q or "loan" in q) and money and rate is not None and years:
        return "emi_calculator", {"loan_amount": money[0], "annual_interest_rate": rate, "tenure_years": years}
    if "budget" in q and money:
        return "budget_planner", {"monthly_income": money[0]}
    if "save" in q and any(w in q for w in ("reach", "target", "goal")) and money and years:
        args = {"target_amount": money[0], "years": years}
        if rate is not None:
            args["annual_return_rate"] = rate
        return "goal_sip_calculator", args
    if ("invest" in q or "sip" in q) and money and rate is not None and years:
        return "sip_calculator", {"monthly_investment": money[0], "annual_return_rate": rate, "years": years}
    return None


class OfflineToolRouter(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "offline-tool-router"

    def bind_tools(self, tools, **kwargs):
        return self                      # routing is hard-wired, nothing to bind

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs) -> ChatResult:
        if isinstance(messages[-1], ToolMessage):
            msg = self._final_reply(messages)
        else:
            question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            choice = choose_tool(question)
            if choice is None:
                msg = AIMessage(content="I can help with loan EMIs, SIP returns, savings goals, "
                                        "50-30-20 budgets and basic calculations. "
                                        "That question is outside what my tools cover.")
            else:
                name, args = choice
                msg = AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                                         "id": "call_1", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @staticmethod
    def _final_reply(messages):
        question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        results = []
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                break
            if isinstance(m, ToolMessage):
                results.insert(0, m.content)
        text = "\n\n".join(results)
        if "Required monthly saving" in text and _rate(question) is None:
            text += "\n\nNote: you did not mention an expected return, so 12% per year was assumed."
        return AIMessage(content=text)
