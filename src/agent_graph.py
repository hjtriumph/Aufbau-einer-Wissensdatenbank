from __future__ import annotations

from typing import TypedDict, List, Any, Dict
import json
import re

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .rag_chain import get_llm, build_retriever


class AgentState(TypedDict, total=False):
    question: str
    docs: List[Document]
    answer: str

    # ReAct control fields
    thought: str
    action: str
    action_input: str
    observation: str

    # trace / memory
    scratchpad: str
    steps: List[Dict[str, str]]
    iterations: int


def _safe_json_loads(text: str) -> dict:
    """
    Robust JSON extraction:
    - strips ```json fences
    - extracts first {...} block if extra text exists
    - returns {} if parsing fails
    """
    if not text:
        return {}
    t = text.strip()

    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    try:
        return json.loads(t)
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    return {}


def _dedupe_docs_keep_order(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []

    for d in docs or []:
        md = d.metadata or {}
        key = (
            md.get("source", ""),
            md.get("page", None),
            md.get("chunk_id", None),
            (d.page_content or "").strip(),
        )
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _format_docs_as_context(docs: List[Document], max_context_chars: int = 12000) -> str:
    parts: List[str] = []
    total = 0

    for d in docs or []:
        md = d.metadata or {}
        src = md.get("source_basename") or md.get("source", "unknown")
        page = md.get("page", None)
        chunk_id = md.get("chunk_id", None)

        header = f"Source: {src}"
        if page is not None:
            header += f" | page={page}"
        if chunk_id is not None:
            header += f" | chunk={chunk_id}"

        block = header + "\n" + (d.page_content or "")
        if total + len(block) > max_context_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts).strip()


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs or []:
        md = d.metadata or {}
        out.append(
            {
                "source": md.get("source", "unknown"),
                "page": md.get("page", None),
                "chunk_id": md.get("chunk_id", None),
            }
        )
    return out


REACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Building Technology agent that follows the ReAct paradigm.

Your job is to answer the user's question by deciding the next action yourself.

You can choose exactly one action each turn:
- "search": search the Building Technology knowledge base
- "finish": stop searching and write the final answer

Output JSON only with exactly these keys:
{
  "thought": "...",
  "action": "search" or "finish",
  "action_input": "..."
}

Rules:
- Use "search" if you still need evidence from the knowledge base.
- Use "finish" only when you already have enough evidence.
- Keep "thought" short and task-focused.
- Keep "action_input" short and useful.
- Do not output any text outside JSON.
- Always answer in the same language as the user's question.
""",
        ),
        (
            "human",
            """Question: {question}

Current scratchpad:
{scratchpad}

Decide the next step now.""",
        ),
    ]
)

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Building Technology assistant.

You MUST answer strictly based on the provided Context and cite evidence.
If the Context is insufficient, clearly output:

INSUFFICIENT_CONTEXT: <reason>

Then still provide a short suggestion for better retrieval keywords.

Output format (keep the section titles exactly as written below):

1) Conclusion
2) Explanation (bullet points)
3) Evidence (source/page/chunk + short summary)
4) Assumptions & Limitations
5) Next Retrieval Suggestions

Rules:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
- Do NOT invent evidence not present in the Context.
- Be concise but specific.
""",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def build_agent_graph(max_steps: int = 4, max_context_chars: int = 12000):
    """
    True ReAct-style Agentic RAG:

      think -> (search or finish)
         search -> observe -> think -> ...
         finish -> final_answer

    Here the LLM explicitly decides whether to search again or finish.
    """
    llm = get_llm()
    retriever = build_retriever()

    def _init_defaults(state: AgentState) -> AgentState:
        return {
            "question": state.get("question", ""),
            "docs": state.get("docs", []),
            "answer": state.get("answer", ""),
            "thought": state.get("thought", ""),
            "action": state.get("action", ""),
            "action_input": state.get("action_input", ""),
            "observation": state.get("observation", ""),
            "scratchpad": state.get("scratchpad", ""),
            "steps": state.get("steps", []),
            "iterations": int(state.get("iterations", 0)),
        }

    def think(state: AgentState) -> AgentState:
        state = _init_defaults(state)

        msg = REACT_PROMPT.format_messages(
            question=state["question"],
            scratchpad=state["scratchpad"] or "(empty)",
        )
        resp = llm.invoke(msg).content
        data = _safe_json_loads(resp)

        thought = str(data.get("thought", "") or "").strip()
        action = str(data.get("action", "") or "").strip().lower()
        action_input = str(data.get("action_input", "") or "").strip()

        if action not in {"search", "finish"}:
            # fallback: prefer search if there is no evidence yet, otherwise finish
            action = "search" if not state.get("docs") else "finish"

        if action == "search" and not action_input:
            action_input = state["question"]

        if action == "finish" and not action_input:
            action_input = "Use the collected evidence to answer the question."

        new_steps = list(state.get("steps", []))
        new_steps.append(
            {
                "thought": thought,
                "action": action,
                "action_input": action_input,
            }
        )

        return {
            **state,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "steps": new_steps,
            "iterations": int(state.get("iterations", 0)) + 1,
        }

    def search(state: AgentState) -> AgentState:
        state = _init_defaults(state)
        query = (state.get("action_input") or "").strip() or state["question"]

        try:
            new_docs = retriever.invoke(query)  # type: ignore
        except Exception:
            new_docs = retriever.get_relevant_documents(query)  # fallback

        all_docs = _dedupe_docs_keep_order(list(state.get("docs", [])) + list(new_docs or []))

        # build short observation text for the scratchpad
        obs_lines: List[str] = []
        for d in (new_docs or [])[:4]:
            md = d.metadata or {}
            src = md.get("source_basename") or md.get("source", "unknown")
            page = md.get("page", None)
            chunk_id = md.get("chunk_id", None)

            header = f"Source: {src}"
            if page is not None:
                header += f" | page={page}"
            if chunk_id is not None:
                header += f" | chunk={chunk_id}"

            snippet = (d.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            obs_lines.append(header + " | " + snippet)

        observation = "\n".join(obs_lines).strip()
        if not observation:
            observation = "No useful documents were retrieved."

        scratchpad = (state.get("scratchpad", "") or "").strip()
        new_block = (
            f"Thought: {state.get('thought', '')}\n"
            f"Action: search\n"
            f"Action Input: {query}\n"
            f"Observation: {observation}\n"
        )
        scratchpad = (scratchpad + "\n\n" + new_block).strip() if scratchpad else new_block

        new_steps = list(state.get("steps", []))
        if new_steps:
            new_steps[-1]["observation"] = observation

        return {
            **state,
            "docs": all_docs,
            "observation": observation,
            "scratchpad": scratchpad,
            "steps": new_steps,
        }

    def final_answer(state: AgentState) -> AgentState:
        state = _init_defaults(state)

        docs = _dedupe_docs_keep_order(state.get("docs", []))
        context = _format_docs_as_context(docs, max_context_chars=max_context_chars)

        msg = FINAL_ANSWER_PROMPT.format_messages(
            question=state["question"],
            context=context,
        )
        ans = llm.invoke(msg).content

        scratchpad = (state.get("scratchpad", "") or "").strip()
        finish_block = (
            f"Thought: {state.get('thought', '')}\n"
            f"Action: finish\n"
            f"Action Input: {state.get('action_input', '')}\n"
        )
        scratchpad = (scratchpad + "\n\n" + finish_block).strip() if scratchpad else finish_block

        return {
            **state,
            "answer": ans,
            "scratchpad": scratchpad,
            "docs": docs,
        }

    def route_after_think(state: AgentState):
        state = _init_defaults(state)

        # force finish when step budget is reached
        if int(state.get("iterations", 0)) >= max_steps:
            return "final_answer"

        if state.get("action") == "search":
            return "search"

        return "final_answer"

    g = StateGraph(AgentState)
    g.add_node("think", think)
    g.add_node("search", search)
    g.add_node("final_answer", final_answer)

    g.set_entry_point("think")
    g.add_conditional_edges(
        "think",
        route_after_think,
        {
            "search": "search",
            "final_answer": "final_answer",
        },
    )
    g.add_edge("search", "think")
    g.add_edge("final_answer", END)

    return g.compile()