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


FEW_SHOT_EXAMPLES = [
    {
        "question": "Does adding occupancy data improve heating and electricity load forecasting in hotels?",
        "context": """Source: Influence of occupancy data on heating and.pdf | page=2
The inclusion of occupancy data slightly reduces the average normalized root mean squared error (NRMSE) for heating forecasts from 4.91 % to 4.90 %, and for electricity forecasts from 7.20 % to 7.07 %. The permutation feature importance (PFI) shows that occupancy contributes up to 14.7 % in heating forecasts.""",
        "answer": """1) Conclusion
Yes, but the improvement is small overall. Occupancy slightly improves average forecasting accuracy, with a somewhat clearer benefit for electricity than for heating.

2) Explanation
- The reported average heating NRMSE decreases from 4.91% to 4.90%.
- The reported average electricity NRMSE decreases from 7.20% to 7.07%.
- The feature importance analysis shows that occupancy can still be relevant in some cases, contributing up to 14.7%.

3) Evidence (source/page/chunk + short summary)
- Source: Influence of occupancy data on heating and.pdf | page=2 | chunk=unknown: Heating NRMSE decreases from 4.91% to 4.90%.
- Source: Influence of occupancy data on heating and.pdf | page=2 | chunk=unknown: Electricity NRMSE decreases from 7.20% to 7.07%.
- Source: Influence of occupancy data on heating and.pdf | page=2 | chunk=unknown: Occupancy PFI reaches up to 14.7%.

4) Assumptions & Scope
- This conclusion is limited to the hotel datasets and forecasting setup described in the provided context.
- The observed gain is small and should not be treated as universally large.

5) Suggested Next Retrieval
occupancy feature importance hotel heating electricity forecasting ED LSTM"""
    },
    {
        "question": "What lagged history is preferable for multi-step heating load forecasting: 24 h or 48 h?",
        "context": """Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=1
By using 24 instead of 48 lagged hours, the simulation time was reduced from 92.75 s to 45.80 s and the forecast accuracy was increased.

Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=16
A lag of 36 or 48 h brings only small advantages compared to 24 h. While the training time more than doubles with 48 instead of 24 lagged hours.""",
        "answer": """1) Conclusion
24 h is preferable in the reported study, because it improves efficiency and can also improve accuracy compared with 48 h.

2) Explanation
- Using 24 lagged hours instead of 48 reduced simulation time from 92.75 s to 45.80 s.
- The study explicitly reports that forecast accuracy increased when using 24 instead of 48 lagged hours.
- Longer histories such as 36 h or 48 h provided only small additional advantages over 24 h.
- The computational cost increased strongly, with training time more than doubling for 48 h.

3) Evidence (source/page/chunk + short summary)
- Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=1 | chunk=unknown: 24 h reduced simulation time from 92.75 s to 45.80 s and increased forecast accuracy.
- Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=16 | chunk=unknown: 36 h and 48 h bring only small advantages over 24 h, while 48 h more than doubles training time.

4) Assumptions & Scope
- This recommendation is tied to the datasets and encoder-decoder forecasting setup reported in the study.
- The best lag length could differ for other buildings or model architectures.

5) Suggested Next Retrieval
lagged hours 24 vs 48 multi-step heating load forecasting encoder decoder"""
    },
    {
        "question": "How does the Mixture of Experts framework organize experts for heat load forecasting?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=5
We assign buildings to experts using a data-driven clustering of their annual mean heat load... We evaluated k in {2, 3, 5}. Empirically, k = 3 achieved the lowest average MAE.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6
We use the outside temperature (T), a historical time series of the heat load (Q˙), and four temporal encodings: sin(Month), cos(Month), sin(Hour), and cos(Hour).""",
        "answer": """1) Conclusion
The framework clusters buildings by heat-load regime and trains one specialized expert per cluster. In the reported setup, k = 3 performed best.

2) Explanation
- Buildings are grouped by a data-driven clustering of annual mean heat load.
- Each cluster is assigned its own expert, so the model can learn regime-specific dynamics.
- The reported experiments compared k = 2, 3, and 5.
- The best average MAE was achieved with k = 3.
- The forecasting inputs include temperature, historical heat load, and sinusoidal month/hour encodings.

3) Evidence (source/page/chunk + short summary)
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=5 | chunk=unknown: Experts are assigned through clustering of annual mean heat load; k = 3 is best.
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6 | chunk=unknown: Inputs are outside temperature, heat-load history, and sin/cos month/hour encodings.

4) Assumptions & Scope
- This answer refers to the specific MoE design described in the provided paper.
- Another dataset or clustering design could lead to a different optimal number of experts.

5) Suggested Next Retrieval
mixture of experts k means annual mean heat load gating expert clustering"""
    },
    {
        "question": "Why is the proposed MoE approach described as metadata-free?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=1
Forecasting across diverse buildings remains challenging because metadata describing these differences (e.g., usage type, occupancy, solar radiation, temperature, climate, HVAC topology) is often unavailable or inconsistent.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=2
A metadata-free approach relying solely on data-driven consumption clustering, without using explicit building metadata.""",
        "answer": """1) Conclusion
The approach is called metadata-free because it does not rely on explicit building metadata and instead organizes buildings using consumption-based clustering.

2) Explanation
- The paper argues that useful metadata is often unavailable or inconsistent across buildings.
- Because of this, the framework avoids depending on fields such as usage type or HVAC topology.
- Instead, it relies on data-driven clustering derived from historical consumption patterns.
- This allows the model to specialize experts without requiring manually curated metadata.

3) Evidence (source/page/chunk + short summary)
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=1 | chunk=unknown: Building metadata is often unavailable or inconsistent.
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=2 | chunk=unknown: The method is explicitly described as metadata-free and based on consumption clustering.

4) Assumptions & Scope
- Metadata-free here means no explicit building metadata is used for the expert design.
- It does not mean the model ignores all structure; it still uses patterns extracted from the historical time series.

5) Suggested Next Retrieval
metadata free mixture of experts building heat load clustering without building metadata"""
    },
    {
        "question": "Which input features are used in the MoE forecasting task?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6
We use the outside temperature (T), a historical time series of the heat load (Q˙), and four temporal encodings: sin(Month), cos(Month), sin(Hour), and cos(Hour).

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6
The outside temperature exhibits a strong negative correlation with Q˙, while monthly encodings capture seasonal variation and hourly encodings represent the day-night cyclic pattern.""",
        "answer": """1) Conclusion
The MoE model uses outside temperature, historical heat load, and four sinusoidal temporal encodings for month and hour.

2) Explanation
- The physical/weather input is outside temperature.
- The core autoregressive signal is the historical heat-load time series.
- Seasonal structure is represented by sin(Month) and cos(Month).
- Daily cyclic structure is represented by sin(Hour) and cos(Hour).
- These encodings help preserve smooth transitions at cycle boundaries.

3) Evidence (source/page/chunk + short summary)
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6 | chunk=unknown: Features are T, historical Q˙, sin(Month), cos(Month), sin(Hour), cos(Hour).
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=6 | chunk=unknown: Temperature is strongly negatively correlated with heat load; month and hour encodings capture seasonal and daily structure.

4) Assumptions & Scope
- This answer describes the feature set reported for the MoE framework in the paper.
- Additional features may be possible in future extensions, but they are not part of the provided context here.

5) Suggested Next Retrieval
MoE feature selection outside temperature historical heat load sin cos month hour"""
    },
    {
        "question": "How is explainability achieved in the MoE framework?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=2
Explainability is provided not only at the MoE level by analysing gating weights that quantify each expert’s contribution to the final prediction, but also at the expert level by inspecting attention patterns and performing feature-importance analysis.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=14
The gating behavior aligns well with the physical characteristics of each building type, providing a transparent view of how forecasting responsibility is distributed.""",
        "answer": """1) Conclusion
Explainability is achieved by analyzing the gating weights, the experts’ attention patterns, and feature importance.

2) Explanation
- Gating weights show how much each expert contributes to the final forecast.
- Attention analysis helps interpret which time steps each expert focuses on.
- Feature-importance analysis clarifies which input variables drive the prediction.
- Together, these views explain both expert routing and expert-specific forecasting behavior.
- The paper also states that the gating behavior aligns with building characteristics, which supports interpretability.

3) Evidence (source/page/chunk + short summary)
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=2 | chunk=unknown: Explainability comes from gating weights, attention patterns, and feature-importance analysis.
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=14 | chunk=unknown: Gating behavior aligns with physical building characteristics and transparently distributes forecasting responsibility.

4) Assumptions & Scope
- This explanation is limited to the interpretability mechanisms described in the provided paper.
- It does not claim full causal explainability; it explains how the model behaves internally.

5) Suggested Next Retrieval
MoE explainability gating weights attention feature importance heat load forecasting"""
    },
    {
        "question": "How much does the MoE improve over the strongest baseline?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=17
Compared to the strongest baseline, the Informer, the MoE reduced NRMSE and MAE by 25.9% and 24.4% on Demandlib, by 12.2% and 12.5% on Green Fusion, and by 10.4% and 10.0% on Hotels.""",
        "answer": """1) Conclusion
The MoE improves over the strongest baseline across all three datasets, with the largest gain on Demandlib and smaller but still positive gains on Green Fusion and Hotels.

2) Explanation
- On Demandlib, the reported reductions are 25.9% in NRMSE and 24.4% in MAE.
- On Green Fusion, the reported reductions are 12.2% in NRMSE and 12.5% in MAE.
- On Hotels, the reported reductions are 10.4% in NRMSE and 10.0% in MAE.
- This indicates that the advantage is consistent across synthetic and real-world settings.

3) Evidence (source/page/chunk + short summary)
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=17 | chunk=unknown: Demandlib improvement is 25.9% NRMSE and 24.4% MAE.
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=17 | chunk=unknown: Green Fusion improvement is 12.2% NRMSE and 12.5% MAE.
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=17 | chunk=unknown: Hotels improvement is 10.4% NRMSE and 10.0% MAE.

4) Assumptions & Scope
- These percentages compare MoE to the strongest baseline identified in the paper, namely the Informer.
- The reported gains are averaged in the paper’s evaluation setting and should be interpreted in that context.

5) Suggested Next Retrieval
MoE strongest baseline Informer percentage improvement NRMSE MAE Demandlib Green Fusion Hotels"""
    },
    {
        "question": "What are the main limitations and future directions of the MoE study?",
        "context": """Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=16
Several limitations should be acknowledged: simple regime design and gating; consumption-based expert specialization without richer metadata; empirical hyperparameter configuration; regional evaluation scope; and offline evaluation without real-time deployment.

Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=16
Future work includes richer regimes and gating, expert specialization with metadata, systematic hyperparameter tuning, cross-region validation, real-time deployment, and multi-energy or multi-task forecasting.""",
        "answer": """1) Conclusion
The study shows strong forecasting results, but it is limited by a simple regime design, restricted metadata usage, empirical tuning, limited evaluation scope, and lack of real-time deployment.

2) Explanation
- Regimes are defined in a relatively simple way and the gate is also simple.
- Expert specialization is mainly based on annual heat demand, without richer metadata.
- Hyperparameters were chosen empirically rather than through a fully systematic search.
- The evaluation is regionally limited and has not yet been tested broadly across different climates.
- The framework is evaluated offline, so operational benefits in real-time control are not yet demonstrated.
- The proposed future directions include richer gating, metadata integration, broader validation, and multi-energy forecasting.

3) Evidence (source/page/chunk + short summary)
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=16 | chunk=unknown: Main limitations include simple regime design, empirical hyperparameters, regional scope, and offline evaluation.
- Source: An_Explainable_Transformer-based_Mixture_of_Expert.pdf | page=16 | chunk=unknown: Future work includes richer regimes, metadata, hyperparameter tuning, cross-region validation, real-time deployment, and multi-task extensions.

4) Assumptions & Scope
- This summary is restricted to the limitation and future-work section provided in the paper.
- It does not imply that the framework is weak; it only states the paper’s own boundaries and next steps.

5) Suggested Next Retrieval
MoE limitations future work metadata gating hyperparameter tuning cross region real time deployment"""
    },
    {
        "question": "What is the role of SHAP values in explainable heating load forecasting?",
        "context": """Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=1
The study combines multi-step heating load forecasting with explainability and uses SHAP values to analyze the contribution of input variables.

Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=16
The explainability analysis helps identify the importance of lagged values and weather-related variables for the forecasting task.""",
        "answer": """1) Conclusion
SHAP values are used to interpret how different input features contribute to the heating load forecast.

2) Explanation
- The forecasting model is not only evaluated by prediction accuracy but also by interpretability.
- SHAP provides a feature-level explanation for the model output.
- This helps identify which lagged heating-load values are influential.
- It also helps assess the contribution of weather-related variables.
- As a result, the method supports both forecasting and model understanding.

3) Evidence (source/page/chunk + short summary)
- Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=1 | chunk=unknown: The study uses SHAP values to analyze the contribution of input variables.
- Source: Explainable multi-step heating load forecasting Using SHAP values and.pdf | page=16 | chunk=unknown: Explainability analysis highlights the importance of lagged values and weather-related variables.

4) Assumptions & Scope
- This answer is limited to the explainability role described in the provided study.
- SHAP explains model behavior, not necessarily physical causality.

5) Suggested Next Retrieval
SHAP heating load forecasting feature contribution lagged values weather variables"""
    },
    {
        "question": "Why are generative methods useful for missing heating-load data imputation?",
        "context": """Source: Imputing the long-term missing heating load data using a generative network.pdf | page=1
Long-term missing heating load data are difficult to recover with simple interpolation because the missing period can span complex seasonal and operational patterns.

Source: Imputing the long-term missing heating load data using a generative network.pdf | page=1
A generative network is proposed to reconstruct missing heating load data by learning realistic temporal patterns from available observations.""",
        "answer": """1) Conclusion
Generative methods are useful because they can reconstruct long missing heating-load segments by learning realistic temporal patterns that simple interpolation may miss.

2) Explanation
- Long gaps in heating-load data are structurally difficult because they can cover seasonal and operational changes.
- Simple interpolation is often too weak for such complex missing periods.
- A generative model can learn broader temporal behavior from the observed data.
- This makes it more suitable for reconstructing realistic missing sequences rather than only smoothing between endpoints.

3) Evidence (source/page/chunk + short summary)
- Source: Imputing the long-term missing heating load data using a generative network.pdf | page=1 | chunk=unknown: Long-term missing heating load is hard to recover with simple interpolation.
- Source: Imputing the long-term missing heating load data using a generative network.pdf | page=1 | chunk=unknown: A generative network is proposed to learn realistic temporal patterns for reconstruction.

4) Assumptions & Scope
- This conclusion refers to long-term missing-data scenarios, not short random gaps.
- The answer does not claim that a generative model is always best in every dataset; it states why it is attractive for this problem setting.

5) Suggested Next Retrieval
generative network long term missing heating load data imputation seasonal patterns"""
    },
]


SYSTEM_TEXT = """You are a Building Technology assistant.

You MUST answer strictly based on the provided Context and cite evidence.
If the Context is insufficient, clearly output:

INSUFFICIENT_CONTEXT: <reason>

Then still provide a short suggestion for better retrieval keywords.

Output format (keep the section titles exactly as written below):

1) Conclusion
2) Explanation (bullet points)
3) Evidence (source/page/chunk + short summary)
4) Assumptions & Scope
5) Suggested Next Retrieval

Rules:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
- Do NOT copy the few-shot examples verbatim.
- Use the examples only as formatting and reasoning guidance.
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

    prompt_messages.append(("human", "Question: {input}\n\nContext:\n{context}"))

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