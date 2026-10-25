"""
finance_tools.py
LangChain tools that wrap the formulas in finance_core.py.
The docstrings matter: the LLM reads them to decide which tool fits a question.
"""
from langchain_core.tools import tool
from finance_core import (inr, emi, sip_future_value, required_monthly_saving,
                          budget_split, safe_eval)


@tool(parse_docstring=True)
def calculator(expression: str) -> str:
    """Evaluate a general arithmetic expression such as percentages, sums or products.
    Use this for plain maths that is not a loan, investment or budget question.

    Args:
        expression: Arithmetic only, e.g. '0.20 * 85000' or '(1200 + 800) / 4'. Write percentages as decimals.
    """
    try:
        value = safe_eval(expression)
    except Exception as err:
        return f"Could not evaluate '{expression}': {err}"
    shown = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
    return f"{expression} = {shown}"


@tool(parse_docstring=True)
def emi_calculator(loan_amount: float, annual_interest_rate: float, tenure_years: float) -> str:
    """Calculate the monthly EMI (equated monthly instalment) of a loan.
    Use this for home, car or personal loan questions.

    Args:
        loan_amount: Loan principal in rupees. 10 lakh = 1000000.
        annual_interest_rate: Yearly interest rate in percent, e.g. 9 for 9%.
        tenure_years: Loan tenure in years.
    """
    try:
        monthly = emi(loan_amount, annual_interest_rate, tenure_years)
    except ValueError as err:
        return f"Invalid input: {err}"
    months = round(tenure_years * 12)
    total = monthly * months
    return (f"Loan {inr(loan_amount)} at {annual_interest_rate}% p.a. for {tenure_years:g} years ({months} months)\n"
            f"Monthly EMI: {inr(monthly)}\n"
            f"Total repayment: {inr(total)}\n"
            f"Total interest paid: {inr(total - loan_amount)}")


@tool(parse_docstring=True)
def sip_calculator(monthly_investment: float, annual_return_rate: float, years: float) -> str:
    """Calculate the future value of a monthly SIP (systematic investment plan).
    Use this when the user invests a fixed amount every month and wants to know the final corpus.

    Args:
        monthly_investment: Amount invested every month in rupees.
        annual_return_rate: Expected yearly return in percent, e.g. 12 for 12%.
        years: Number of years the SIP runs.
    """
    try:
        fv = sip_future_value(monthly_investment, annual_return_rate, years)
    except ValueError as err:
        return f"Invalid input: {err}"
    months = round(years * 12)
    invested = monthly_investment * months
    return (f"SIP of {inr(monthly_investment)}/month at {annual_return_rate}% p.a. for {years:g} years ({months} months)\n"
            f"Total invested: {inr(invested)}\n"
            f"Estimated returns: {inr(fv - invested)}\n"
            f"Future value: {inr(fv)}")


@tool(parse_docstring=True)
def goal_sip_calculator(target_amount: float, years: float, annual_return_rate: float = 12.0) -> str:
    """Work out how much must be saved every month to reach a target amount.
    Use this for goal questions like 'how much should I save monthly to reach X in Y years'.
    If the user gives no return rate, the default of 12% is used and the answer must mention that assumption.

    Args:
        target_amount: Amount the user wants to have at the end, in rupees. 20 lakh = 2000000.
        years: Years available to reach the target.
        annual_return_rate: Expected yearly return in percent. Defaults to 12.
    """
    try:
        monthly = required_monthly_saving(target_amount, annual_return_rate, years)
    except ValueError as err:
        return f"Invalid input: {err}"
    months = round(years * 12)
    return (f"Target {inr(target_amount)} in {years:g} years at an assumed {annual_return_rate}% p.a. return\n"
            f"Required monthly saving: {inr(monthly)}\n"
            f"Total contribution over {months} months: {inr(monthly * months)}")


@tool(parse_docstring=True)
def budget_planner(monthly_income: float) -> str:
    """Suggest a monthly budget using the 50-30-20 rule (needs, wants, savings).
    Use this when the user asks for a budget or spending plan for a given income.

    Args:
        monthly_income: Monthly take-home income in rupees.
    """
    try:
        split = budget_split(monthly_income)
    except ValueError as err:
        return f"Invalid input: {err}"
    lines = [f"Monthly income: {inr(monthly_income)} (50-30-20 rule)"]
    for name, share in (("Needs", 50), ("Wants", 30), ("Savings", 20)):
        lines.append(f"{name} ({share}%): {inr(split[name])}")
    return "\n".join(lines)


TOOLS = [calculator, emi_calculator, sip_calculator, goal_sip_calculator, budget_planner]
