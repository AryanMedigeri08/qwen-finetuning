import csv
import json
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from filelock import FileLock

from sahayak import ROOT, RUNTIME
from sahayak.db import DEFAULT_DB, Store
from sahayak.graph import Assistant
from sahayak.model import LocalQwen, Qwen
from sahayak.retrieval import LexicalSearch, PolicySearch
from sahayak.tools import Tools

PAGES = ("Chat", "Trace viewer", "Experiments", "Database", "About")
DEMOS = {
    "Refund timing": "How long does a UPI refund take?",
    "Track an order": "Where is order KK-10001?",
    "Cancel an order": "Please cancel order KK-10001",
}

st.set_page_config(page_title="Sahayak | KharidiKart Support", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f7f9f7; }
    .block-container { max-width: 1120px; padding-top: 2.5rem; }
    [data-testid="stSidebar"] { background: #ebf3ee; }
    [data-testid="stChatMessage"] {
      background: white; border: 1px solid #dce6de; border-radius: .8rem; margin: .4rem 0;
    }
    .stButton button { border-radius: .6rem; }
    @media (max-width: 650px) { .block-container { padding: 1.75rem 1rem; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_store(path):
    store = Store(path)
    store.seed()
    return store


@st.cache_resource
def get_retrieval(demo):
    return LexicalSearch() if demo else PolicySearch()


@st.cache_resource
def get_model():
    if endpoint := os.environ.get("SAHAYAK_LOCAL_SERVER"):
        return LocalQwen(endpoint)
    return Qwen(nf4=os.environ.get("SAHAYAK_NF4") == "1")


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def reset(customer):
    st.session_state.owner = customer
    st.session_state.session = uuid4().hex
    st.session_state.history = []
    st.session_state.context = {}
    st.session_state.last = {}


def submit(query, customer, store, demo):
    try:
        with st.spinner("Checking your request…"):
            model = None if demo else get_model()
            assistant = Assistant(Tools(store, get_retrieval(demo)), model=model)
            result = assistant.ask(customer, query, st.session_state.context, st.session_state.history)
    except (OSError, RuntimeError, ImportError) as error:
        st.error(f"The support runtime could not load. {error}")
        return
    st.session_state.context = result["next_context"]
    st.session_state.history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": result["answer"]}])
    st.session_state.last = result
    path = RUNTIME / "chat-traces.jsonl"
    event = {"session": st.session_state.session, "customer": customer, "mode": "demo" if demo else "C4",
             "query": query, "intent": result.get("intent"), "trace": result["trace"], "steps": result["steps"]}
    with FileLock(str(path) + ".lock"), path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(event, ensure_ascii=False) + "\n")
    st.rerun()


def show_answer(text):
    st.text(text)
    from re import findall
    for filename, number in dict.fromkeys(findall(r"\[([\w-]+\.md) #(\d+)\]", text)):
        if (ROOT / "policies" / filename).is_file():
            with st.expander(f"Source · {filename} §{number}"):
                headings = (ROOT / "policies" / filename).read_text(encoding="utf-8").split("\n## ")
                if int(number) < len(headings):
                    st.markdown("#### " + headings[int(number)])


def chat(customer, store, demo):
    st.title("How can we help?")
    if not st.session_state.history:
        st.write("Ask about an order or a store policy.")
        if customer == "C0001":
            for col, (label, query) in zip(st.columns(3), DEMOS.items()):
                if col.button(label, use_container_width=True):
                    submit(query, customer, store, demo)
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            show_answer(item["content"])
    if st.session_state.context.get("kind") == "confirm":
        args = st.session_state.context["arguments"]
        if "new_address" in args:
            st.text(f'New address: {args["new_address"]}')
        left, right = st.columns(2)
        if left.button("Confirm action", type="primary", use_container_width=True):
            submit("yes", customer, store, demo)
        if right.button("Keep order unchanged", use_container_width=True):
            submit("no", customer, store, demo)
    if query := st.chat_input("Ask KharidiKart support", max_chars=2000):
        submit(query, customer, store, demo)


def traces():
    st.title("Trace viewer")
    result = st.session_state.last
    if not result:
        st.info("Ask a question in Chat to see its trace.")
        return
    left, middle, right = st.columns(3)
    left.metric("Graph steps", result["steps"])
    middle.metric("Tool calls", len(result["calls"]))
    right.metric("Verifier", result.get("verdict") or "—")
    if "confidence" in result:
        st.caption(f'Router · {result["intent"]} · confidence {result["confidence"]:.2f}')
        if result.get("alternatives"):
            st.table([{"Alternative": label, "First-token probability": round(value, 3)}
                      for label, value in result["alternatives"]])
    for event in result["trace"]:
        with st.expander(f'{event["node"].title()} · {event["ms"]:,.0f} ms', expanded=True):
            st.json(event)


def experiments():
    st.title("Experiments")
    results = ROOT / "results"
    st.subheader("Scenario evaluation")
    live_rows = read_rows(results / "main_table.csv")
    if live_rows:
        st.dataframe(live_rows, hide_index=True, use_container_width=True)
    else:
        st.info("The C0–C5 comparison appears after a complete matched run.")
    cpu_rows = read_rows(results / "cpu_scenarios.csv")
    if cpu_rows:
        st.subheader("Exploratory CPU comparison")
        fields = ("config", "precision", "n", "tsr_pct", "tool_accuracy_pct", "hallucination_flags_pct",
                  "placeholder_pct", "n_red", "safe_handling_pct")
        st.dataframe([{key: row.get(key) for key in fields} for row in cpu_rows], hide_index=True, use_container_width=True)
    figures = [
        ("fig1_main.png", "Task success and grounding flags"),
        ("fig3_data_budget.png", "Data budget"),
        ("fig4_router_confusion.png", "Router confusion matrix"),
        ("fig5_latency.png", "Turn latency"),
    ]
    for name, label in figures:
        if (ROOT / "figures" / name).is_file():
            st.image(str(ROOT / "figures" / name), caption=label)
    st.subheader("QLoRA language evaluation")
    st.dataframe(read_rows(results / "training_notebook_e1.csv"), hide_index=True, use_container_width=True)
    if (results / "retrieval.json").is_file():
        st.subheader("Policy retrieval")
        retrieval = json.loads((results / "retrieval.json").read_text(encoding="utf-8"))
        left, middle, right = st.columns(3)
        left.metric("Questions", retrieval["n"])
        middle.metric("Recall@3", f'{retrieval["recall_at_3"]:.1%}')
        right.metric("MRR@3", f'{retrieval["mrr_at_3"]:.3f}')


def database(customer, store):
    st.title("Your demo records")
    orders, refunds = store.customer_records(customer)
    st.subheader("Orders")
    st.dataframe(orders, hide_index=True, use_container_width=True)
    st.subheader("Refunds")
    if refunds:
        st.dataframe(refunds, hide_index=True, use_container_width=True)
    else:
        st.info("This demo customer has no refunds.")


def about():
    st.title("About SahayakLLM")
    st.write("Team 10 built this research prototype for the fictional KharidiKart storefront.")
    st.write("The workflow combines Qwen2.5-1.5B-Instruct and a QLoRA adapter with four agents, a policy index, and a mock order database.")
    st.write("Demo accounts and transactions are fictional. The account picker is for demonstrations.")
    st.markdown("[Qwen2.5-1.5B-Instruct · Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)")
    st.markdown("[SahayakLLM adapter](https://huggingface.co/aryan22medigeri/llm-qwen2.5-1.5b-qlora)")


def main():
    demo = os.environ.get("SAHAYAK_DEMO") == "1"
    store = get_store(Path(os.environ.get("SAHAYAK_DB", DEFAULT_DB)))
    st.sidebar.title("KharidiKart")
    st.sidebar.caption("Sahayak support workspace")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    customers = {row["customer_id"]: row for row in store.customers()}
    customer = st.sidebar.selectbox("Demo customer", list(customers), format_func=lambda value: f'{value} · {customers[value]["city"]}')
    st.sidebar.caption("Offline demo mode" if demo else "Qwen + LangGraph")
    if st.session_state.get("owner") != customer:
        reset(customer)
    if page == "Chat":
        chat(customer, store, demo)
    elif page == "Trace viewer":
        traces()
    elif page == "Experiments":
        experiments()
    elif page == "Database":
        database(customer, store)
    else:
        about()
    st.caption("Dataset · Bitext Customer Support LLM Chatbot Training Dataset (Bitext Innovations) · CDLA-Sharing-1.0")


main()
