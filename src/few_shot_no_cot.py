from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from .rag_chain import get_llm, build_retriever


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


def _format_docs_as_context(docs: List[Document]) -> str:
    parts: List[str] = []

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

        parts.append(header + "\n" + (d.page_content or ""))

    return "\n\n---\n\n".join(parts).strip()


# No-reasoning few-shot examples:
# - short direct answers only
# - no explanation
# - no evidence section
# - no step-by-step reasoning
FEW_SHOT_EXAMPLES = [
    {
        "question": "Does adding occupancy data improve heating and electricity load forecasting in hotels?",
        "context": """Source: Influence of occupancy data on heating and.pdf | page=2
The inclusion of occupancy data slightly reduces the average normalized root mean squared error (NRMSE) for heating forecasts from 4.91 % to 4.90 %, and for electricity forecasts from 7.20 % to 7.07 %. The permutation feature importance (PFI) shows that occupancy contributes up to 14.7 % in heating forecasts.""",
        "answer": "Yes. It improves both heating and electricity load forecasting slightly, but the overall gain is small."
    },
    {
        "question": "What lagged history is preferable for multi-step heating load forecasting: 24 h or 48 h?",
        "context": """Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=1
By using 24 instead of 48 lagged hours, the simulation time was reduced from 92.75 s to 45.80 s and the forecast accuracy was increased.

Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=16
A lag of 36 or 48 h brings only small advantages compared to 24 h. While the training time more than doubles with 48 instead of 24 lagged hours.""",
        "answer": "24 h is preferable because it improves forecast accuracy and reduces computation time compared with 48 h."
    },
    {
        "question": "How does the Mixture of Experts framework organize experts for heat load forecasting?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=5
We assign buildings to experts using a data-driven clustering of their annual mean heat load... We evaluated k in {2, 3, 5}. Empirically, k = 3 achieved the lowest average MAE.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6
We use the outside temperature (T), a historical time series of the heat load (Q˙), and four temporal encodings: sin(Month), cos(Month), sin(Hour), and cos(Hour).""",
        "answer": "The framework clusters buildings by annual mean heat load and assigns one expert to each cluster. In the reported study, k = 3 performed best."
    },
    {
        "question": "Why is the proposed MoE approach described as metadata-free?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=1
Forecasting across diverse buildings remains challenging because metadata describing these differences (e.g., usage type, occupancy, solar radiation, temperature, climate, HVAC topology) is often unavailable or inconsistent.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=2
A metadata-free approach relying solely on data-driven consumption clustering, without using explicit building metadata.""",
        "answer": "It is described as metadata-free because it relies on data-driven consumption clustering instead of explicit building metadata."
    },
    {
        "question": "Which input features are used in the MoE forecasting task?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6
We use the outside temperature (T), a historical time series of the heat load (Q˙), and four temporal encodings: sin(Month), cos(Month), sin(Hour), and cos(Hour).

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6
The outside temperature exhibits a strong negative correlation with Q˙, while monthly encodings capture seasonal variation and hourly encodings represent the day-night cyclic pattern.""",
        "answer": "The inputs are outside temperature, historical heat load, sin(Month), cos(Month), sin(Hour), and cos(Hour)."
    },
    {
        "question": "How is explainability achieved in the MoE framework?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=2
Explainability is provided not only at the MoE level by analysing gating weights that quantify each expert’s contribution to the final prediction, but also at the expert level by inspecting attention patterns and performing feature-importance analysis.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=14
The gating behavior aligns well with the physical characteristics of each building type, providing a transparent view of how forecasting responsibility is distributed.""",
        "answer": "Explainability is achieved through gating weights, attention patterns, and feature-importance analysis."
    },
    {
        "question": "How much does the MoE improve over the strongest baseline?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=17
Compared to the strongest baseline, the Informer, the MoE reduced NRMSE and MAE by 25.9% and 24.4% on Demandlib, by 12.2% and 12.5% on Green Fusion, and by 10.4% and 10.0% on Hotels.""",
        "answer": "The MoE improves over the strongest baseline on all three datasets, with the largest gain on Demandlib."
    },
    {
        "question": "What are the main limitations and future directions of the MoE study?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=16
Several limitations should be acknowledged: simple regime design and gating; consumption-based expert specialization without richer metadata; empirical hyperparameter configuration; regional evaluation scope; and offline evaluation without real-time deployment.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=16
Future work includes richer regimes and gating, expert specialization with metadata, systematic hyperparameter tuning, cross-region validation, real-time deployment, and multi-energy or multi-task forecasting.""",
        "answer": "The main limitations are simple regime design, limited metadata usage, empirical tuning, restricted evaluation scope, and no real-time deployment. Future work includes richer gating, metadata integration, broader validation, and real-time extensions."
    },
    {
        "question": "What is the role of SHAP values in explainable heating load forecasting?",
        "context": """Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=1
The study combines multi-step heating load forecasting with explainability and uses SHAP values to analyze the contribution of input variables.

Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=16
The explainability analysis helps identify the importance of lagged values and weather-related variables for the forecasting task.""",
        "answer": "SHAP values are used to interpret how input features contribute to the heating load forecast."
    },
    {
        "question": "Why are generative methods useful for missing heating-load data imputation?",
        "context": """Source: Imputing the long-term missing heating load data using a generative network.pdf | page=1
Long-term missing heating load data are difficult to recover with simple interpolation because the missing period can span complex seasonal and operational patterns.

Source: Imputing the long-term missing heating load data using a generative network.pdf | page=1
A generative network is proposed to reconstruct missing heating load data by learning realistic temporal patterns from available observations.""",
        "answer": "Generative methods are useful because they can reconstruct long missing heating-load segments by learning realistic temporal patterns that simple interpolation may miss."
    },
]


SYSTEM_TEXT = """You are a Building Technology assistant.

You MUST answer strictly based on the provided Context.

Rules:
- Give a short, direct answer only.
- Do NOT provide step-by-step reasoning.
- Do NOT provide an Explanation section.
- Do NOT provide an Evidence section.
- Do NOT provide Assumptions, Scope, or Retrieval Suggestions.
- Do NOT reveal intermediate reasoning.
- Do NOT cite sources explicitly in the answer.
- If the Context is insufficient, clearly output:
  INSUFFICIENT_CONTEXT: <reason>

Language rule:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
- Do NOT copy the few-shot examples verbatim.
- Use the examples only to learn the short-answer style.
"""


def build_few_shot_rag_chain(num_examples: int = 3):
    llm = get_llm()
    retriever = build_retriever()
    parser = StrOutputParser()

    selected_examples = FEW_SHOT_EXAMPLES[:max(0, num_examples)]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Question: {question}\n\nContext:\n{context}"),
            ("ai", "{answer}"),
        ]
    )

    prompt_messages = [("system", SYSTEM_TEXT)]

    if selected_examples:
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=selected_examples,
            example_prompt=example_prompt,
        )
        prompt_messages.extend(few_shot_prompt.format_messages())

    prompt_messages.append(
        ("human", "Question: {input}\n\nContext:\n{context}")
    )

    final_prompt = ChatPromptTemplate.from_messages(prompt_messages)

    def _retrieve(x: Dict[str, Any]) -> Dict[str, Any]:
        q = str(x.get("input", "")).strip()
        docs = retriever.invoke(q)
        return {
            "input": q,
            "context_docs": docs,
            "context": _format_docs_as_context(docs),
        }

    def _generate(x: Dict[str, Any]) -> Dict[str, Any]:
        answer = (final_prompt | llm | parser).invoke(
            {
                "input": x["input"],
                "context": x["context"],
            }
        )
        return {
            "answer": answer,
            "context": x["context"],
            "sources": extract_sources(x["context_docs"]),
            "num_examples": num_examples,
        }

    return RunnableLambda(_retrieve) | RunnableLambda(_generate)