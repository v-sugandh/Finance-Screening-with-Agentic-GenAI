"""
finance_core.py
Plain Python maths for the personal finance agent.
No LangChain in here on purpose, so every formula can be tested on its own.
"""
import ast
import operator


# ---------- formatting ----------
def inr(amount, decimals=2):
    """Format a number in Indian digit grouping, e.g. 1234567.5 -> ₹12,34,567.50"""
    sign = "-" if amount < 0 else ""
    whole, _, frac = f"{abs(amount):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if decimals else "")


# ---------- input checks ----------
def _check(value, name, allow_zero=False):
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'zero or ' if allow_zero else ''}greater than zero (got {value})")


def monthly_rate(annual_percent):
    """Annual percentage -> monthly decimal rate (12% -> 0.01)."""
    return annual_percent / 12 / 100


# ---------- the formulas from the assignment brief ----------
def emi(principal, annual_rate, years):
    """EMI = P * r * (1+r)^n / ((1+r)^n - 1)"""
    _check(principal, "Loan amount")
    _check(annual_rate, "Interest rate", allow_zero=True)
    _check(years, "Tenure")
    n = round(years * 12)
    r = monthly_rate(annual_rate)
    if r == 0:                      # interest free loan, formula would divide by zero
        return principal / n
    growth = (1 + r) ** n
    return principal * r * growth / (growth - 1)


def sip_future_value(monthly_amount, annual_rate, years):
    """FV = P * ((1+r)^n - 1) / r   (instalment paid at the end of each month)"""
    _check(monthly_amount, "Monthly investment")
    _check(annual_rate, "Expected return", allow_zero=True)
    _check(years, "Investment period")
    n = round(years * 12)
    r = monthly_rate(annual_rate)
    if r == 0:
        return monthly_amount * n
    return monthly_amount * ((1 + r) ** n - 1) / r


def required_monthly_saving(target, annual_rate, years):
    """The SIP formula turned around: P = FV * r / ((1+r)^n - 1)"""
    _check(target, "Target amount")
    _check(annual_rate, "Expected return", allow_zero=True)
    _check(years, "Time horizon")
    n = round(years * 12)
    r = monthly_rate(annual_rate)
    if r == 0:
        return target / n
    return target * r / ((1 + r) ** n - 1)


def budget_split(income):
    """50-30-20 rule."""
    _check(income, "Monthly income")
    return {"Needs": income * 0.50, "Wants": income * 0.30, "Savings": income * 0.20}


# ---------- calculator ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def safe_eval(expression):
    """
    Evaluate an arithmetic expression without using eval().
    Only numbers, + - * / // % ** and brackets are accepted.
    """
    cleaned = (expression.replace(",", "").replace("₹", "").replace("×", "*")
               .replace("÷", "/").replace("^", "**").strip())
    tree = ast.parse(cleaned, mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("Only plain arithmetic is allowed")

    return walk(tree)
