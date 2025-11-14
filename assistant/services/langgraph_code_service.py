# assistant/services/langgraph_code_service.py
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

# ---- LAZY CLIENT CREATION ----
_client = None  # <-- no error at import time

def get_client() -> OpenAI:
    """Create the OpenAI client only when it's actually needed."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # raise a clear error *only when we actually call the assistant*
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Set it in your environment before calling the code assistant."
            )
        _client = OpenAI(api_key=api_key)
    return _client


class AssistantState(TypedDict, total=False):
    user_input: str
    intent: str
    examples: list
    llm_result: str


def classify_intent(state: AssistantState) -> AssistantState:
    text = state["user_input"].lower()
    if "explain" in text or "what does" in text or "شرح" in text:
        state["intent"] = "explain_code"
    else:
        state["intent"] = "generate_code"
    return state


def retrieve_examples(state: AssistantState) -> AssistantState:
    # TODO: plug your example logic here if you want
    state["examples"] = []
    return state


def generate_code_node(state: AssistantState) -> AssistantState:
    client = get_client()  # <-- use lazy client
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate clean Python code, return it fenced in ```python```.",
            },
            {"role": "user", "content": state["user_input"]},
        ],
        temperature=0.2,
    )
    state["llm_result"] = completion.choices[0].message.content
    return state


def explain_code_node(state: AssistantState) -> AssistantState:
    client = get_client()  # <-- use lazy client
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Explain the Python code step by step.",
            },
            {"role": "user", "content": state["user_input"]},
        ],
        temperature=0.0,
    )
    state["llm_result"] = completion.choices[0].message.content
    return state


def route_after_retrieval(state: AssistantState) -> str:
    return state.get("intent", "generate_code")


def build_graph():
    builder = StateGraph(AssistantState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_examples", retrieve_examples)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("explain_code", explain_code_node)

    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "retrieve_examples")

    builder.add_conditional_edges(
        "retrieve_examples",
        route_after_retrieval,
        {
            "generate_code": "generate_code",
            "explain_code": "explain_code",
        },
    )

    builder.add_edge("generate_code", END)
    builder.add_edge("explain_code", END)

    return builder.compile()


graph = build_graph()


def run_code_assistant(user_input: str) -> str:
    state = graph.invoke({"user_input": user_input})
    return state.get("llm_result", "")
