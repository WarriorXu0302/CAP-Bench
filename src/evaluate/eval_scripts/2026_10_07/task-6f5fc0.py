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
TASK_ID = "task-6f5fc0"
TASK_DESCRIPTION = 'I’m writing a report on global dengue transmission trends and need your help with some research. First, go to Google Scholar and search for epidemiological studies on dengue from the past year. Find several papers with relatively high citation counts, prioritizing information that can be obtained directly from titles and abstracts, and identify the high-risk regions and transmission patterns mentioned in those papers. If a result requires institutional login or payment to access the full text, note that access restriction and move on to another paper with publicly available abstract information.\n\nThen, go to Our World in Data and check global dengue data to see which countries or regions have had the fastest growth in case numbers in recent years, and whether that aligns with the high-risk areas identified in the papers. If a map view is available, that would be even better for visualizing geographic distribution.\n\nFinally, help me summarize whether the conclusions from academic research are consistent with actual data trends, and highlight any new findings or contradictions.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ScholarPapers(BaseModel):
    """Papers extracted from Google Scholar search results"""
    papers_found: Optional[List[str]] = Field(default_factory=list)
    high_risk_regions: Optional[List[str]] = Field(default_factory=list)
    transmission_patterns: Optional[List[str]] = Field(default_factory=list)
    time_filter_applied: Optional[bool] = None
    sorted_by_citations: Optional[bool] = None


class OWIDData(BaseModel):
    """Our World in Data dengue information"""
    fastest_growth_countries: Optional[List[str]] = Field(default_factory=list)
    map_view_used: Optional[bool] = None
    data_trends_mentioned: Optional[str] = None


class Synthesis(BaseModel):
    """Synthesis of academic vs actual data"""
    consistency_assessment: Optional[str] = None
    new_findings_or_contradictions: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_scholar_papers() -> str:
    return """
Extract information about Google Scholar search results from the answer:

- papers_found: list of paper titles or identifiers mentioned
- high_risk_regions: list of high-risk regions or geographic areas mentioned in the papers
- transmission_patterns: list of transmission patterns or epidemiological findings mentioned
- time_filter_applied: true if the answer indicates filtering for past year, false otherwise
- sorted_by_citations: true if the answer indicates sorting by citation count, false otherwise

If any field is missing or not mentioned, set appropriate defaults (empty lists or null).
"""


def prompt_extract_owid_data() -> str:
    return """
Extract information about Our World in Data dengue data from the answer:

- fastest_growth_countries: list of countries or regions identified as having fastest growth in dengue cases
- map_view_used: true if the answer mentions using or viewing a map visualization, false otherwise
- data_trends_mentioned: any description of data trends observed

If any field is missing, set to null or empty list as appropriate.
"""


def prompt_extract_synthesis() -> str:
    return """
Extract the synthesis or comparison from the answer:

- consistency_assessment: the answer's assessment of whether academic research aligns with actual data trends
- new_findings_or_contradictions: any new findings or contradictions highlighted

If not present, set to null.
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


def mentions_google_scholar(text: Optional[str]) -> bool:
    return has_any_ci(text, ['google scholar', 'scholar.google'])


def mentions_dengue(text: Optional[str]) -> bool:
    return has_any_ci(text, ['dengue', '登革热'])


def mentions_time_filter(text: Optional[str]) -> bool:
    return has_any_ci(text, ['past year', 'last year', 'recent', '最近一年', '去年'])


def mentions_citations(text: Optional[str]) -> bool:
    return has_any_ci(text, ['citation', 'cited', '引用', '被引'])


def mentions_owid(text: Optional[str]) -> bool:
    return has_any_ci(text, ['our world in data', 'owid', 'ourworldindata'])


def mentions_map_view(text: Optional[str]) -> bool:
    return has_any_ci(text, ['map', 'geographic', 'distribution', '地图', '分布'])


def mentions_growth_or_trend(text: Optional[str]) -> bool:
    return has_any_ci(text, ['growth', 'increase', 'trend', 'fastest', '增长', '趋势', '最快'])


def has_regions_list(regions: Optional[List[str]]) -> bool:
    return bool(regions and len(regions) > 0)


def has_papers_list(papers: Optional[List[str]]) -> bool:
    return bool(papers and len(papers) > 1)


def mentions_consistency(text: Optional[str]) -> bool:
    return has_any_ci(text, ['consistent', 'align', 'match', 'agree', 'correspond', '一致', '吻合', '符合'])


def mentions_abstract_or_title(text: Optional[str]) -> bool:
    return has_any_ci(text, ['abstract', 'title', '摘要', '标题'])


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
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()

    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
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
    scholar_info = await evaluator.extract(
        prompt=prompt_extract_scholar_papers(),
        template_class=ScholarPapers,
        extraction_name="scholar_papers"
    )

    owid_info = await evaluator.extract(
        prompt=prompt_extract_owid_data(),
        template_class=OWIDData,
        extraction_name="owid_data"
    )

    synthesis_info = await evaluator.extract(
        prompt=prompt_extract_synthesis(),
        template_class=Synthesis,
        extraction_name="synthesis"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Google Scholar section
    scholar_node = evaluator.add_sequential(
        id="google_scholar_section",
        desc="Google Scholar search for dengue epidemiological studies",
        parent=root,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A1 - Academic search
    scholar_search_ok = (mentions_google_scholar(answer) and
                        mentions_dengue(answer))
    evaluator.add_custom_node(
        result=bool(scholar_search_ok),
        id="scholar_action_search",
        desc="[Action Node] scholar.google.com:F1:A1 - Search for dengue epidemiological studies on Google Scholar",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A6 - Time filtering
    time_filter_ok = (scholar_info.time_filter_applied or
                     mentions_time_filter(answer))
    evaluator.add_custom_node(
        result=bool(time_filter_ok),
        id="scholar_action_time_filter",
        desc="[Action Node] scholar.google.com:F3:A6 - Apply time filter to show papers from the past year",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A7 - Sort by citations
    sort_citations_ok = (scholar_info.sorted_by_citations or
                        mentions_citations(answer))
    evaluator.add_custom_node(
        result=bool(sort_citations_ok),
        id="scholar_action_sort_citations",
        desc="[Action Node] scholar.google.com:F3:A7 - Sort results by citation count",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F4:P3 - Identify citation counts
    citation_recognition_ok = mentions_citations(answer)
    evaluator.add_custom_node(
        result=bool(citation_recognition_ok),
        id="scholar_perception_citation_recognition",
        desc="[Perception Node] scholar.google.com:F4:P3 - Recognize and compare papers by citation counts",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P1 - Understand paper content
    content_understanding_ok = (has_regions_list(scholar_info.high_risk_regions) or
                               has_papers_list(scholar_info.papers_found) or
                               mentions_abstract_or_title(answer))
    evaluator.add_custom_node(
        result=bool(content_understanding_ok),
        id="scholar_perception_content_understanding",
        desc="[Perception Node] scholar.google.com:F1:P1 - Extract high-risk regions and transmission patterns from paper titles and abstracts",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A15 - Pagination
    multiple_papers_ok = has_papers_list(scholar_info.papers_found)
    evaluator.add_custom_node(
        result=bool(multiple_papers_ok),
        id="scholar_action_pagination",
        desc="[Action Node] scholar.google.com:F1:A15 - Browse through multiple pages to find several papers",
        parent=scholar_node,
        critical=False
    )

    # 3.2 Our World in Data section
    owid_node = evaluator.add_sequential(
        id="owid_section",
        desc="Our World in Data - Global dengue data analysis",
        parent=root,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A1 - Data search
    owid_search_ok = (mentions_owid(answer) and
                     mentions_dengue(answer))
    evaluator.add_custom_node(
        result=bool(owid_search_ok),
        id="owid_action_search",
        desc="[Action Node] ourworldindata.org:F1:A1 - Search for and locate dengue data on Our World in Data",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F2:A1 - Timeline manipulation
    timeline_ok = (mentions_time_filter(answer) or
                  has_any_ci(answer, ['recent years', 'years', '近几年', '多年']))
    evaluator.add_custom_node(
        result=bool(timeline_ok),
        id="owid_action_timeline",
        desc="[Action Node] ourworldindata.org:F2:A1 - Manipulate timeline to observe data changes over recent years",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P1 - Trend identification
    trend_identification_ok = (has_regions_list(owid_info.fastest_growth_countries) or
                              mentions_growth_or_trend(answer))
    evaluator.add_custom_node(
        result=bool(trend_identification_ok),
        id="owid_perception_trend_identification",
        desc="[Perception Node] ourworldindata.org:F1:P1 - Identify countries/regions with fastest case growth from trend lines",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F4:A4 - View switching
    view_switch_ok = (owid_info.map_view_used or
                     mentions_map_view(answer))
    evaluator.add_custom_node(
        result=bool(view_switch_ok),
        id="owid_action_view_switch",
        desc="[Action Node] ourworldindata.org:F4:A4 - Switch to map view for geographic visualization",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F4:P4 - Map distribution awareness
    map_perception_ok = (mentions_map_view(answer) and
                        has_any_ci(answer, ['distribution', 'color', 'geographic', '分布', '颜色', '地理']))
    evaluator.add_custom_node(
        result=bool(map_perception_ok),
        id="owid_perception_map_distribution",
        desc="[Perception Node] ourworldindata.org:F4:P4 - Interpret geographic distribution from map color intensity",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A8 - Data point interaction
    data_interaction_ok = (has_regions_list(owid_info.fastest_growth_countries) or
                          has_any_ci(answer, ['country', 'region', 'specific', '国家', '地区', '具体']))
    evaluator.add_custom_node(
        result=bool(data_interaction_ok),
        id="owid_action_data_interaction",
        desc="[Action Node] ourworldindata.org:F1:A8 - Interact with data points to view specific country values",
        parent=owid_node,
        critical=False
    )

    # 3.3 Synthesis section
    synthesis_node = evaluator.add_sequential(
        id="synthesis_section",
        desc="Cross-validation and synthesis of academic research vs actual data",
        parent=root,
        critical=False
    )

    # Check if synthesis attempts to compare the two data sources
    synthesis_comparison_ok = (bool(synthesis_info.consistency_assessment) or
                              mentions_consistency(answer))
    evaluator.add_custom_node(
        result=bool(synthesis_comparison_ok),
        id="synthesis_comparison",
        desc="Compare and assess consistency between academic research conclusions and actual data trends",
        parent=synthesis_node,
        critical=False
    )

    # Check if new findings or contradictions are highlighted
    findings_highlighted_ok = (bool(synthesis_info.new_findings_or_contradictions) or
                              has_any_ci(answer, ['finding', 'contradiction', 'difference', 'discrepancy', '发现', '矛盾', '差异']))
    evaluator.add_custom_node(
        result=bool(findings_highlighted_ok),
        id="synthesis_findings",
        desc="Highlight new findings or contradictions between sources",
        parent=synthesis_node,
        critical=False
    )

    # Check if regions from both sources are cross-referenced
    regions_from_scholar = has_regions_list(scholar_info.high_risk_regions)
    regions_from_owid = has_regions_list(owid_info.fastest_growth_countries)
    cross_reference_ok = regions_from_scholar and regions_from_owid
    evaluator.add_custom_node(
        result=bool(cross_reference_ok),
        id="synthesis_cross_reference",
        desc="Cross-reference high-risk regions from papers with actual growth data",
        parent=synthesis_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
