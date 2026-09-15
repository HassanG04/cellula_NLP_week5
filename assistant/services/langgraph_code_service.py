from __future__ import annotations

import ast
import logging
import os
import re
import time
from functools import lru_cache
from typing import TypedDict

logger = logging.getLogger(__name__)


class AssistantState(TypedDict, total=False):
    user_input: str
    intent: str
    llm_result: str
    syntax_valid: bool


def classify_intent(state: AssistantState) -> AssistantState:
    text = state["user_input"].lower()
    return {
        **state,
        "intent": "explain_code"
        if any(word in text for word in ["explain", "what does", "شرح"])
        else "generate_code",
    }


def validate_generated_python(text: str) -> bool:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return bool(code.strip())


@lru_cache
def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    from openai import OpenAI

    return OpenAI(timeout=30, max_retries=0)


def _complete(state, explain):
    started = time.perf_counter()
    completion = get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {
                "role": "system",
                "content": "Explain Python code step by step."
                if explain
                else "Generate Python code in a fenced python block. Never execute code.",
            },
            {"role": "user", "content": state["user_input"]},
        ],
        max_completion_tokens=500,
    )
    content = completion.choices[0].message.content or ""
    if not content.strip() or (not explain and not validate_generated_python(content)):
        raise RuntimeError("Model returned invalid Python output")
    logger.info(
        "code_assistant intent=%s duration_ms=%.3f tokens=%s",
        state["intent"],
        (time.perf_counter() - started) * 1000,
        completion.usage.total_tokens if completion.usage else None,
    )
    return {**state, "llm_result": content, "syntax_valid": not explain}


def generate_code_node(state):
    return _complete(state, False)


def explain_code_node(state):
    return _complete(state, True)


@lru_cache
def build_graph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AssistantState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("explain_code", explain_code_node)
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        lambda state: state["intent"],
        {"generate_code": "generate_code", "explain_code": "explain_code"},
    )
    builder.add_edge("generate_code", END)
    builder.add_edge("explain_code", END)
    return builder.compile()


def run_code_assistant(user_input):
    if not isinstance(user_input, str) or not user_input.strip() or len(user_input) > 10000:
        raise ValueError("input must be a nonempty string of at most 10000 characters")
    return build_graph().invoke({"user_input": user_input})["llm_result"]
