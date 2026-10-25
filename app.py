"""
FinWise AI — Personal Finance Assistant
Run with: streamlit run app.py

The interface is presentation-focused. Core agent behavior remains in agent_setup.py.
"""
import os
import streamlit as st

from agent_setup import build_agent, run_query

st.set_page_config(
    page_title="FinWise AI | Personal Finance",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    ("Loan planning", "Calculate EMI for ₹10 lakh loan at 8% interest for 20 years"),
    ("Investment growth", "If I invest ₹7000 monthly for 15 years at 10% return"),
    ("Monthly budget", "Create a budget plan for income ₹90,000"),
    ("Quick calculation", "What is 20% of ₹85,000"),
    ("SIP estimate", "How much will I get if I invest ₹3000 per month for 5 years at 8% return"),
    ("Savings goal", "How much should I save monthly to reach ₹20 lakh in 10 years?"),
]

# Load configured API keys from Streamlit secrets, falling back to environment variables.
secrets_loaded = False
for key_name in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
    try:
        secret_value = st.secrets.get(key_name)
    except Exception:
        secret_value = None
    if secret_value:
        os.environ[key_name] = secret_value
        secrets_loaded = True

# Google's library warns when both GEMINI_API_KEY and GOOGLE_API_KEY are set, so drop the former.
os.environ.pop("GEMINI_API_KEY", None)

def friendly_error(err):
    """Short, readable reason for a failed request. The raw text is shown separately."""
    text = str(err)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "The AI service's free daily quota has been used up. Try again after it resets, or use another API key."
    if "503" in text or "UNAVAILABLE" in text:
        return "The AI service is busy right now. Wait a minute and ask again."
    if any(w in text for w in ("API key", "credentials", "PERMISSION_DENIED", "401", "403")):
        return "The API key is missing or was rejected. Check .streamlit/secrets.toml."
    return "Something went wrong while answering."


def show_answer(turn):
    """Draw one assistant reply: the answer or the error, plus the tools that were used."""
    if turn.get("error"):
        st.error(turn["answer"])
        with st.expander("Technical details"):
            st.code(turn["error"])
        return
    st.markdown(turn["answer"])
    if turn.get("tools"):
        with st.expander("View calculation tools used"):
            for tool_name, args in turn["tools"]:
                formatted_args = ", ".join(f"{name}={value}" for name, value in args.items())
                st.code(f"{tool_name}({formatted_args})", language="python")


if "history" not in st.session_state:
    st.session_state.history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

st.markdown(
    """
    <style>
    :root { --fw-green: #176b52; --fw-deep: #123d32; --fw-mint: #e9f4ee;
            --fw-ink: #1e2b27; --fw-muted: #66736d; --fw-line: #dce5df; }
    .stApp { background: #f6f8f6; color: var(--fw-ink); }
    [data-testid="stHeader"] { background: rgba(246,248,246,.92); }
    [data-testid="stSidebar"] { background: #eef3ef; border-right: 1px solid var(--fw-line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #43534b; }
    .hero {
        padding: 1.7rem 1.9rem; border-radius: 18px;
        background: linear-gradient(115deg, #123d32 0%, #176b52 70%, #2e8063 100%);
        color: #fff; margin: .35rem 0 1.25rem 0;
        box-shadow: 0 10px 26px rgba(18,61,50,.12);
    }
    .hero-kicker { font-size: .76rem; text-transform: uppercase; letter-spacing: .16em;
                   color: #c6e5d5; font-weight: 700; margin-bottom: .55rem; }
    .hero h1 { color: #fff; font-size: clamp(1.9rem, 4vw, 2.65rem); line-height: 1.12;
               letter-spacing: -.035em; margin: 0 0 .65rem 0; }
    .hero p { color: #e0eee7; max-width: 760px; margin: 0; font-size: 1rem; }
    .section-label { color: var(--fw-green); font-size: .76rem; font-weight: 800;
                     text-transform: uppercase; letter-spacing: .12em; margin-bottom: .25rem; }
    .panel {
        background: #fff; border: 1px solid var(--fw-line); border-radius: 14px;
        padding: 1rem 1.1rem; margin: .35rem 0 .8rem 0;
    }
    .panel-title { font-weight: 750; color: var(--fw-ink); margin-bottom: .25rem; }
    .muted { color: var(--fw-muted); font-size: .9rem; }
    div.stButton > button {
        border-radius: 9px; border: 1px solid #d4e1d8; min-height: 2.6rem;
        font-weight: 600; transition: all .15s ease;
    }
    div.stButton > button:hover { border-color: var(--fw-green); color: var(--fw-green); }
    div[data-testid="stChatMessage"] { border: 1px solid #e2e9e4; border-radius: 14px;
                                      padding: .85rem 1rem; background: #fff; }
    [data-testid="stChatInput"] { border-color: #cbd9d0; }
    .footer { color: #718078; font-size: .78rem; text-align: center;
              padding: 1.3rem 0 .4rem 0; }
    @media (max-width: 700px) {
        .hero { padding: 1.25rem; border-radius: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🌿 FinWise AI")
    st.caption("Your personal finance workspace")
    st.divider()

    with st.expander("Configuration", expanded=not secrets_loaded):
        if secrets_loaded:
            st.success("API key loaded from Streamlit secrets.")
        else:
            st.caption("Enter your provider key, or configure it in `.streamlit/secrets.toml`.")
            entered_key = st.text_input(
                "Google API key",
                type="password",
                key="google_api_key_input",
                help="The key is used for this app session. Avoid sharing it.",
            )
            if entered_key.strip():
                os.environ["GOOGLE_API_KEY"] = entered_key.strip()

    st.markdown("### Start with a question")
    st.caption("Choose a prompt to send it to the assistant.")
    for index, (label, prompt) in enumerate(EXAMPLES):
        if st.button(label, key=f"example_{index}", use_container_width=True):
            st.session_state.pending_question = prompt

    st.divider()
    left, right = st.columns(2)
    with left:
        st.metric("Questions", len(st.session_state.history))
    with right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.history = []
            st.session_state.pending_question = ""
            st.rerun()

@st.cache_resource(show_spinner=False)
def get_agent(api_key_fingerprint):
    # The key fingerprint is a cache argument so changing keys refreshes the agent.
    return build_agent()

key_fingerprint = (
    os.getenv("GOOGLE_API_KEY", "") + "|" + os.getenv("OPENAI_API_KEY", "")
)
agent, model_label = get_agent(key_fingerprint)

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Personal finance • Simplified</div>
      <h1>Make your next money decision<br>with more confidence.</h1>
      <p>Explore loan repayments, SIP projections, savings goals and monthly budgets
      with an AI assistant that can use calculation tools to work through your questions.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

offline = "offline" in model_label
status_text = "● Demo mode" if offline else "● Ready to help"
meta_col, info_col = st.columns([1, 2])
with meta_col:
    st.markdown(
        f'<div class="panel"><div class="section-label">Assistant status</div>'
        f'<div class="panel-title">{status_text}</div>'
        f'<div class="muted">Model: {model_label}</div></div>',
        unsafe_allow_html=True,
    )
with info_col:
    st.markdown(
        '<div class="panel"><div class="section-label">What you can do</div>'
        '<div class="panel-title">Plan · Calculate · Compare</div>'
        '<div class="muted">Ask a question in your own words, or choose a suggested prompt '
        'from the sidebar to get started.</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label">Your conversation</div>', unsafe_allow_html=True)

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        show_answer(turn)

question = st.chat_input("Ask about an EMI, SIP, budget or savings goal…")
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Working through your question…"):
            try:
                result = run_query(agent, question)
                turn = {"question": question,
                        "answer": result.get("answer") or "I couldn't generate an answer for that question.",
                        "tools": result.get("tools_used", [])}
            except Exception as err:
                turn = {"question": question, "answer": friendly_error(err),
                        "tools": [], "error": str(err)[:800]}
        show_answer(turn)

    st.session_state.history.append(turn)
    st.rerun()

st.markdown(
    '<div class="footer">FinWise AI provides educational estimates, not personalized '
    'financial, tax or investment advice. Verify assumptions before making financial decisions.</div>',
    unsafe_allow_html=True,
)
