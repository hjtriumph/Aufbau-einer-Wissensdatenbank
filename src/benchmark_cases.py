from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidenceRef:
    source_substring: str
    page: Optional[int] = None


@dataclass
class Case:
    id: str
    question: str
    question_type: str
    difficulty: str
    gold_answer: str
    key_points: List[str] = field(default_factory=list)
    expected_evidence: List[EvidenceRef] = field(default_factory=list)
    notes: str = ""


TEST_CASES: List[Case] = [
    Case(
        id="case_001",
        question="Does adding occupancy data improve heating and electricity load forecasting in hotels? Give the main conclusion and the reported NRMSE numbers.",
        question_type="evidence_grounded_conclusion",
        difficulty="easy",
        gold_answer=(
            "Yes, but only slightly overall. In the ED model, heating NRMSE improves from 4.91% to 4.90%, "
            "and electricity NRMSE improves from 7.20% to 7.07% when occupancy is included."
        ),
        key_points=[
            "slightly",
            "heating",
            "4.91",
            "4.90",
            "electricity",
            "7.20",
            "7.07",
            "ED model",
        ],
        expected_evidence=[
            EvidenceRef("Influence of occupancy data on heating", 2),
        ],
        notes="Occupancy paper: main abstract result.",
    ),
    Case(
        id="case_002",
        question="According to the occupancy study, how large can the permutation feature importance of occupancy become in heating forecasts?",
        question_type="fact_extraction",
        difficulty="easy",
        gold_answer=(
            "The permutation feature importance of occupancy can contribute up to 14.7% in heating forecasts."
        ),
        key_points=[
            "14.7",
            "occupancy",
            "heating",
            "feature importance",
        ],
        expected_evidence=[
            EvidenceRef("Influence of occupancy data on heating", 2),
            EvidenceRef("Influence of occupancy data on heating", 5),
        ],
        notes="Occupancy paper: abstract + table result.",
    ),
    Case(
        id="case_003",
        question="In the multi-step heating load forecasting study, is 24 hours or 48 hours of lagged history preferable, and why?",
        question_type="numerical_comparison",
        difficulty="easy",
        gold_answer=(
            "24 hours is preferable in the reported study because it reduced simulation time from 92.75 s to 45.80 s "
            "and also increased forecast accuracy compared with 48 hours."
        ),
        key_points=[
            "24",
            "48",
            "45.80",
            "92.75",
            "reduced simulation time",
            "increased forecast accuracy",
        ],
        expected_evidence=[
            EvidenceRef("Explainable multi-step heating load forecasting", 1),
        ],
        notes="Heating forecasting + SHAP paper: highlight/abstract result.",
    ),
    Case(
        id="case_004",
        question="What benefit did Deep SHAP based feature selection provide in the multi-step heating load forecasting study?",
        question_type="fact_extraction",
        difficulty="medium",
        gold_answer=(
            "Deep SHAP based feature selection reduced training time by 3.98% and reduced NRMSE by 8.11%."
        ),
        key_points=[
            "Deep SHAP",
            "3.98",
            "training time",
            "8.11",
            "NRMSE",
        ],
        expected_evidence=[
            EvidenceRef("Explainable multi-step heating load forecasting", 1),
            EvidenceRef("Explainable multi-step heating load forecasting", 2),
        ],
        notes="Heating forecasting + SHAP paper: abstract result.",
    ),
    Case(
        id="case_005",
        question="Why is the proposed Mixture of Experts framework described as metadata-free?",
        question_type="concept_explanation",
        difficulty="easy",
        gold_answer=(
            "It is described as metadata-free because it relies only on data-driven consumption clustering "
            "and does not require explicit building metadata."
        ),
        key_points=[
            "metadata-free",
            "data-driven",
            "consumption clustering",
            "without explicit building metadata",
        ],
        expected_evidence=[
            EvidenceRef("An_Explainable_Transformer-based_Mixture_of_Expert", 1),
            EvidenceRef("An_Explainable_Transformer-based_Mixture_of_Expert", 2),
        ],
        notes="MoE paper: abstract + contribution bullets.",
    ),
    Case(
        id="case_006",
        question="How are experts organized in the explainable Mixture of Experts framework for heat load forecasting?",
        question_type="method_explanation",
        difficulty="medium",
        gold_answer=(
            "Buildings are clustered into heat load regimes using k-means on annual mean heat load, "
            "and one Informer-based expert is trained for each regime. The main model uses three experts."
        ),
        key_points=[
            "clustered",
            "k-means",
            "annual mean heat load",
            "Informer",
            "three experts",
            "k = 3",
        ],
        expected_evidence=[
            EvidenceRef("An_Explainable_Transformer-based_Mixture_of_Expert", 5),
        ],
        notes="MoE paper: expert clustering section.",
    ),
    Case(
        id="case_007",
        question="What input features are used in the Mixture of Experts forecasting task?",
        question_type="fact_extraction",
        difficulty="easy",
        gold_answer=(
            "The input features are outside temperature, historical heat load, and four temporal encodings: "
            "sin(Month), cos(Month), sin(Hour), and cos(Hour)."
        ),
        key_points=[
            "outside temperature",
            "historical heat load",
            "sin(month)",
            "cos(month)",
            "sin(hour)",
            "cos(hour)",
        ],
        expected_evidence=[
            EvidenceRef("An_Explainable_Transformer-based_Mixture_of_Expert", 6),
        ],
        notes="MoE paper: feature selection section.",
    ),
    Case(
        id="case_008",
        question="How does the gating mechanism combine experts in the Mixture of Experts framework?",
        question_type="method_explanation",
        difficulty="medium",
        gold_answer=(
            "The gating unit outputs softmax weights for the experts at each timestep, and the final forecast is a weighted average "
            "of the low-, medium-, and high-load expert predictions."
        ),
        key_points=[
            "softmax",
            "weights",
            "weighted average",
            "low",
            "medium",
            "high",
        ],
        expected_evidence=[
            EvidenceRef("An_Explainable_Transformer-based_Mixture_of_Expert", 6),
        ],
        notes="MoE paper: gating mechanism and inference.",
    ),
    Case(
        id="case_009",
        question="What overall improvements does the Mixture of Experts model report across the Demandlib, Green Fusion, and Hotels datasets?",
        question_type="numerical_comparison",
        difficulty="medium",
        gold_answer=(
            "Averaged over three input windows and three prediction horizons, the MoE reduces NRMSE by 24% on Demandlib, "
            "13% on Green Fusion, and 10% on Hotels."
        ),
        key_points=[
            "24",
            "13",
            "10",
            "Demandlib",
            "Green Fusion",
            "Hotels",
            "NRMSE",
        ],
        expected_evidence=[
            EvidenceRef("An_Explainable_Transformer-based_Mixture_of_Expert", 1),
        ],
        notes="MoE paper: abstract result.",
    ),
    Case(
        id="case_010",
        question="What is the main idea of the generative network for imputing long-term missing heating load data?",
        question_type="concept_explanation",
        difficulty="medium",
        gold_answer=(
            "The main idea is to impute long-term missing heating load data using only contemporaneous weather and temporal data, "
            "without relying on previously imputed heating values, thereby avoiding recursive error accumulation."
        ),
        key_points=[
            "weather",
            "temporal data",
            "without previous imputed values",
            "avoid error accumulation",
            "long-term missing",
        ],
        expected_evidence=[
            EvidenceRef("Imputing the long-term missing heating load data", 1),
            EvidenceRef("Imputing the long-term missing heating load data", 2),
        ],
        notes="Imputation paper: abstract + intro.",
    ),
]