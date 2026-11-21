import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-58a3d0"
TASK_DESCRIPTION = "I'm a long-time follower of Tesla and recently learned that their autonomous driving technology (FSD/Autopilot) is under considerable regulatory scrutiny.\n\nFirst, please navigate to the official SEC website to find Tesla's latest annual report (10-K). In the 'Risk Factors' section, carefully identify and extract the key sentences where the company officially describes risks related to 'regulatory or legal proceedings concerning autonomous driving technology'.\n\nNext, take this official statement and search on Quora for a highly engaged, relevant question (e.g., with numerous answers or followers) to see if the general public or industry professionals are discussing similar concerns.\n\nFinally, please compare: are there any significant discrepancies between the official phrasing in the financial report and the central topics of discussion among Quora users?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SECRiskExtraction(BaseModel):
    """Risk factors extracted from Tesla 10-K regarding autonomous driving"""
    risk_statement: Optional[str] = None
    mentions_regulatory: Optional[bool] = None
    mentions_legal: Optional[bool] = None
    mentions_autonomous_driving: Optional[bool] = None


class QuoraDiscussion(BaseModel):
    """Quora discussion details extracted from the answer"""
    question_title: Optional[str] = None
    engagement_metric: Optional[str] = None
    central_concerns: Optional[str] = None


class DiscrepancyAnalysis(BaseModel):
    """Comparison analysis between SEC and Quora"""
    has_comparison: Optional[bool] = None
    identified_discrepancies: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_sec_risk_from_answer() -> str:
    return """
Extract the Tesla 10-K Risk Factors information related to autonomous driving from the answer.

Return:
- risk_statement: the exact key sentences or statement extracted from the Risk Factors section regarding autonomous driving regulatory or legal risks. If not present, set null.
- mentions_regulatory: true if the extracted statement mentions regulatory aspects, false otherwise.
- mentions_legal: true if the extracted statement mentions legal proceedings or lawsuits, false otherwise.
- mentions_autonomous_driving: true if autonomous driving, FSD, or Autopilot are mentioned, false otherwise.

If any field cannot be determined, set it to null or false as appropriate.
"""


def prompt_extract_quora_from_answer() -> str:
    return """
From the answer, extract details about the Quora search and discussion found:

- question_title: the title or topic of the Quora question mentioned. If not present, set null.
- engagement_metric: any mention of engagement indicators like number of answers, followers, or upvotes. If not present, set null.
- central_concerns: a summary of what Quora users are discussing regarding Tesla autonomous driving. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_discrepancy_analysis_from_answer() -> str:
    return """
From the answer, extract the comparison analysis between SEC filing and Quora discussion:

- has_comparison: true if the answer performs a comparison or analysis between the two sources, false otherwise.
- identified_discrepancies: any specific discrepancies, differences, or gaps identified between official SEC language and Quora discussions. If not present, set null.

If any field is missing, set it to null or false as appropriate.
"""


# --------------------------------------------------------------------------- #
# Helper functions for lenient checks                                         #
# --------------------------------------------------------------------------- #
def ci_contains(text: Optional[str], substr: str) -> bool:
    if not text:
        return False
    return substr.lower() in text.lower()


def has_any_ci(text: Optional[str], substrs: List[str]) -> bool:
    return any(ci_contains(text, s) for s in substrs)


def contains_digits(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'\d', text))


def looks_like_sec_edgar_search(answer: str) -> bool:
    """Check if answer mentions navigating to SEC EDGAR for Tesla"""
    return (has_any_ci(answer, ['sec', 'edgar', 'sec.gov']) and
            has_any_ci(answer, ['tesla']))


def looks_like_10k_filter(answer: str) -> bool:
    """Check if answer mentions filtering or selecting 10-K filings"""
    return has_any_ci(answer, ['10-k', '10k', 'annual report'])


def looks_like_risk_factors_section(answer: str) -> bool:
    """Check if answer mentions locating Risk Factors section"""
    return has_any_ci(answer, ['risk factors', 'risk factor'])


def looks_like_autonomous_driving_extract(risk_statement: Optional[str]) -> bool:
    """Check if extracted risk statement relates to autonomous driving"""
    if not risk_statement:
        return False
    return has_any_ci(risk_statement, ['autonomous', 'autopilot', 'fsd', 'self-driving', 'full self-driving'])


def looks_like_regulatory_legal_extract(risk_statement: Optional[str]) -> bool:
    """Check if extracted risk statement mentions regulatory or legal aspects"""
    if not risk_statement:
        return False
    return (has_any_ci(risk_statement, ['regulatory', 'regulation', 'legal', 'lawsuit', 'litigation', 'proceeding', 'investigation']))


def looks_like_quora_search(answer: str) -> bool:
    """Check if answer mentions searching Quora"""
    return has_any_ci(answer, ['quora'])


def looks_like_engagement_filtering(answer: str) -> bool:
    """Check if answer mentions selecting high-engagement questions"""
    engagement_terms = ['answers', 'followers', 'upvotes', 'views', 'engagement', 'popular', 'highly engaged', 'high engagement']
    return has_any_ci(answer, engagement_terms)


def has_comparison_analysis(answer: str) -> bool:
    """Check if answer performs comparison between SEC and Quora"""
    comparison_terms = ['compare', 'comparison', 'discrepancy', 'discrepancies', 'difference', 'differences',
                       'contrast', 'versus', 'vs', 'gap', 'diverge', 'mismatch']
    return has_any_ci(answer, comparison_terms)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: Any,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    Restrict evaluator.verify to at most one usage.
    Favor lenient, fault-tolerant checks and allow partial credit.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()

    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # -------- 2. Extract information from the answer --------------------- #
    sec_info = await evaluator.extract(
        prompt=prompt_extract_sec_risk_from_answer(),
        template_class=SECRiskExtraction,
        extraction_name="sec_risk_extraction"
    )

    quora_info = await evaluator.extract(
        prompt=prompt_extract_quora_from_answer(),
        template_class=QuoraDiscussion,
        extraction_name="quora_discussion"
    )

    discrepancy_info = await evaluator.extract(
        prompt=prompt_extract_discrepancy_analysis_from_answer(),
        template_class=DiscrepancyAnalysis,
        extraction_name="discrepancy_analysis"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 SEC EDGAR section
    sec_section = evaluator.add_sequential(
        id="sec_edgar_section",
        desc="Navigate SEC EDGAR and extract Tesla 10-K Risk Factors on autonomous driving",
        parent=root,
        critical=False
    )

    # [Action Node] sec.govedgar:F1:A1 - Precise search for Tesla on SEC
    sec_search_ok = looks_like_sec_edgar_search(answer)
    evaluator.add_custom_node(
        result=bool(sec_search_ok),
        id="sec_action_search_tesla",
        desc="[Action Node] sec.govedgar:F1:A1 - Navigate to SEC EDGAR and search for Tesla",
        parent=sec_section,
        critical=False
    )

    # [Action Node] sec.govedgar:F1:A19 - Filter for 10-K filing type
    tenk_filter_ok = looks_like_10k_filter(answer)
    evaluator.add_custom_node(
        result=bool(tenk_filter_ok),
        id="sec_action_filter_10k",
        desc="[Action Node] sec.govedgar:F1:A19 - Filter or select the latest annual report (10-K) filing",
        parent=sec_section,
        critical=False
    )

    # [Perception Node] sec.govedgar:F4:P11 - Long document understanding: locate and extract Risk Factors
    risk_factors_located = looks_like_risk_factors_section(answer)
    risk_statement_valid = bool(sec_info and sec_info.risk_statement and len(sec_info.risk_statement.strip()) > 20)
    autonomous_driving_mentioned = looks_like_autonomous_driving_extract(sec_info.risk_statement)
    regulatory_legal_mentioned = looks_like_regulatory_legal_extract(sec_info.risk_statement)

    long_doc_understanding_ok = (risk_factors_located and risk_statement_valid and
                                autonomous_driving_mentioned and regulatory_legal_mentioned)

    evaluator.add_custom_node(
        result=bool(long_doc_understanding_ok),
        id="sec_perception_risk_factors",
        desc="[Perception Node] sec.govedgar:F4:P11 - Locate Risk Factors section in 10-K and extract key sentences about autonomous driving regulatory/legal risks",
        parent=sec_section,
        critical=False
    )

    # 3.2 Quora section
    quora_section = evaluator.add_sequential(
        id="quora_section",
        desc="Search Quora for high-engagement discussions on Tesla autonomous driving concerns",
        parent=root,
        critical=False
    )

    # [Action Node] quora.com:F1:A1 - Cross-modal search: convert SEC legal terminology to natural language
    quora_search_ok = looks_like_quora_search(answer)
    has_sec_context = bool(sec_info and sec_info.risk_statement)
    cross_modal_search_ok = quora_search_ok and has_sec_context

    evaluator.add_custom_node(
        result=bool(cross_modal_search_ok),
        id="quora_action_cross_modal_search",
        desc="[Action Node] quora.com:F1:A1 - Transform SEC legal terminology into natural language keywords and search Quora for relevant discussions",
        parent=quora_section,
        critical=False
    )

    # [Perception Node] quora.com:F1:P2 - Data awareness: identify high-engagement questions
    engagement_filtering_ok = looks_like_engagement_filtering(answer)
    engagement_metric_present = bool(quora_info and quora_info.engagement_metric)
    question_identified = bool(quora_info and quora_info.question_title)

    data_perception_ok = engagement_filtering_ok and (engagement_metric_present or question_identified)

    evaluator.add_custom_node(
        result=bool(data_perception_ok),
        id="quora_perception_engagement",
        desc="[Perception Node] quora.com:F1:P2 - Identify and select high-engagement question based on metadata (answers, followers, etc.)",
        parent=quora_section,
        critical=False
    )

    # Additional check: extract central concerns from Quora discussion
    quora_concerns_extracted = bool(quora_info and quora_info.central_concerns)
    evaluator.add_custom_node(
        result=bool(quora_concerns_extracted),
        id="quora_extract_concerns",
        desc="Extract central concerns and topics from Quora discussion",
        parent=quora_section,
        critical=False
    )

    # 3.3 Comparison and analysis section
    comparison_section = evaluator.add_sequential(
        id="comparison_section",
        desc="Compare SEC official language with Quora public discussion and identify discrepancies",
        parent=root,
        critical=False
    )

    # Check if comparison was performed
    comparison_performed = has_comparison_analysis(answer)
    has_both_sources = bool(sec_info and sec_info.risk_statement and quora_info and quora_info.central_concerns)

    evaluator.add_custom_node(
        result=bool(comparison_performed and has_both_sources),
        id="comparison_performed",
        desc="Perform semantic comparison between SEC official statement and Quora discussion topics",
        parent=comparison_section,
        critical=False
    )

    # Check if discrepancies were identified
    discrepancies_identified = bool(discrepancy_info and discrepancy_info.identified_discrepancies)

    evaluator.add_custom_node(
        result=bool(discrepancies_identified),
        id="discrepancies_identified",
        desc="Identify and explain any significant discrepancies between official SEC language and public Quora discussions",
        parent=comparison_section,
        critical=False
    )

    # Final synthesis check
    provides_conclusion = (comparison_performed and
                          (discrepancies_identified or
                           has_any_ci(answer, ['no significant discrepancy', 'no major difference', 'aligned', 'consistent', 'similar'])))

    evaluator.add_custom_node(
        result=bool(provides_conclusion),
        id="final_conclusion",
        desc="Provide a clear conclusion about the comparison between SEC and Quora sources",
        parent=comparison_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
