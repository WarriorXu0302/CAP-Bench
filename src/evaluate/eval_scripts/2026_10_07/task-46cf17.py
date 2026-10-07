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
TASK_ID = "task-46cf17"
TASK_DESCRIPTION = "I've recently been researching 3D reconstruction techniques related to Neural Radiance Fields (NeRF), and I'd like to reproduce experiments from a representative paper.\n\nPlease help me search arXiv for papers with titles containing 'Neural Radiance Fields' or 'NeRF', filtering for those published since January 2024 and categorized under Computer Vision (cs.CV). From these, identify 3 seemingly important papers (based on their titles and abstracts), and record their arXiv ID, title, authors, and publication date.\n\nThen, search for these 3 papers individually on Semantic Scholar. For each paper, check its TLDR summary, total citation count, and highly influential citation count. Select the paper with the highest total citation count as the target paper.\n\nFor this target paper, expand to view its complete citation statistics. Switch to the 'Citations' tab, filter for 'Highly Influential' citations, and from these, identify 3 subsequent works that also have a relatively high citation count (indicating that the target paper indeed had a significant impact).\n\nFinally, output: the target paper's arXiv ID, title, authors, publication date, arXiv link, TLDR summary, total citation count, highly influential citation count, and Semantic Scholar link; along with the titles, citation counts, and Semantic Scholar links for the 3 highly influential subsequent works."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TargetPaperInfo(BaseModel):
    """Target paper information extracted from the answer"""
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    publication_date: Optional[str] = None
    arxiv_link: Optional[str] = None
    tldr_summary: Optional[str] = None
    total_citation_count: Optional[str] = None
    highly_influential_citation_count: Optional[str] = None
    semantic_scholar_link: Optional[str] = None


class SubsequentWork(BaseModel):
    """Subsequent work information"""
    title: Optional[str] = None
    citation_count: Optional[str] = None
    semantic_scholar_link: Optional[str] = None


class SubsequentWorks(BaseModel):
    """All three subsequent works"""
    work1: Optional[SubsequentWork] = None
    work2: Optional[SubsequentWork] = None
    work3: Optional[SubsequentWork] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_target_paper() -> str:
    return """
Extract the target paper information from the answer. The target paper is the one with the highest total citation count among the 3 papers found.

Return:
- arxiv_id: the arXiv ID exactly as stated
- title: the paper title
- authors: the author names or author list
- publication_date: the publication date
- arxiv_link: the arXiv link URL
- tldr_summary: the TLDR summary from Semantic Scholar
- total_citation_count: the total citation count as a string
- highly_influential_citation_count: the highly influential citation count as a string
- semantic_scholar_link: the Semantic Scholar link URL

If any field is missing, set it to null.
"""


def prompt_extract_subsequent_works() -> str:
    return """
Extract information about the 3 subsequent works (highly influential citations with high citation counts) from the answer.

Return three works (work1, work2, work3), each containing:
- title: the paper title
- citation_count: the citation count as a string
- semantic_scholar_link: the Semantic Scholar link URL

If any field is missing, set it to null.
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


def looks_like_arxiv_id(text: Optional[str]) -> bool:
    if not text:
        return False
    # arXiv ID format: YYMM.NNNNN or YYMM.NNNNNN
    return bool(re.search(r'\d{4}\.\d{4,5}', text))


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def looks_like_date_2024_or_later(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check if contains 2024, 2025, 2026, etc.
    year_match = re.search(r'20(2[4-9]|[3-9]\d)', text)
    return bool(year_match)


def extract_numeric(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Extract first number found
    m = re.search(r'\d+', text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def has_nerf_keywords(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['neural radiance fields', 'nerf'])


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
    target_paper = await evaluator.extract(
        prompt=prompt_extract_target_paper(),
        template_class=TargetPaperInfo,
        extraction_name="target_paper_info"
    )

    subsequent_works = await evaluator.extract(
        prompt=prompt_extract_subsequent_works(),
        template_class=SubsequentWorks,
        extraction_name="subsequent_works_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 arXiv search and filtering
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="arXiv search for NeRF papers with filtering and metadata extraction",
        parent=root,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A2 - Search with title keywords
    search_action_ok = (has_any_ci(answer, ['arxiv']) and
                       has_nerf_keywords(answer))
    evaluator.add_custom_node(
        result=bool(search_action_ok),
        id="arxiv_search_action",
        desc="[Action Node] arxiv.org:F1:A2 - Search arXiv for papers with 'Neural Radiance Fields' or 'NeRF' in title",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A3 - Date range filtering
    title_has_nerf = has_nerf_keywords(target_paper.title)
    date_is_2024_or_later = looks_like_date_2024_or_later(target_paper.publication_date)
    evaluator.add_custom_node(
        result=bool(date_is_2024_or_later),
        id="arxiv_date_filter",
        desc="[Action Node] arxiv.org:F1:A3 - Filter for papers published since January 2024",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F1:P11 - Identify important papers
    has_three_papers_mentioned = has_any_ci(answer, ['3 papers', 'three papers']) or answer.count('arXiv ID') >= 3
    evaluator.add_custom_node(
        result=bool(has_three_papers_mentioned and title_has_nerf),
        id="arxiv_identify_important",
        desc="[Perception Node] arxiv.org:F1:P11 - Identify 3 important papers based on titles and abstracts",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F3:A6 - Click into paper details
    has_arxiv_id = looks_like_arxiv_id(target_paper.arxiv_id)
    evaluator.add_custom_node(
        result=bool(has_arxiv_id),
        id="arxiv_detail_action",
        desc="[Action Node] arxiv.org:F3:A6 - Access paper detail pages to extract metadata",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P4 - Extract author information
    has_authors = bool(target_paper.authors and target_paper.authors.strip())
    evaluator.add_custom_node(
        result=bool(has_authors),
        id="arxiv_author_extraction",
        desc="[Perception Node] arxiv.org:F3:P4 - Extract author names from paper details",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P7 - Extract publication date
    has_pub_date = bool(target_paper.publication_date and target_paper.publication_date.strip())
    evaluator.add_custom_node(
        result=bool(has_pub_date and date_is_2024_or_later),
        id="arxiv_date_extraction",
        desc="[Perception Node] arxiv.org:F3:P7 - Extract publication date from version history",
        parent=arxiv_node,
        critical=False
    )

    # Additional check: arXiv link provided
    has_arxiv_link = looks_like_url(target_paper.arxiv_link) and ci_contains(target_paper.arxiv_link, 'arxiv')
    evaluator.add_custom_node(
        result=bool(has_arxiv_link),
        id="arxiv_link_provided",
        desc="Target paper arXiv link is provided",
        parent=arxiv_node,
        critical=False
    )

    # 3.2 Semantic Scholar search and citation analysis
    semantic_scholar_node = evaluator.add_sequential(
        id="semantic_scholar_section",
        desc="Semantic Scholar paper search and citation analysis",
        parent=root,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A5 - Click into paper details
    has_semantic_link = looks_like_url(target_paper.semantic_scholar_link) and ci_contains(target_paper.semantic_scholar_link, 'semanticscholar')
    semantic_search_ok = has_any_ci(answer, ['semantic scholar']) and has_semantic_link
    evaluator.add_custom_node(
        result=bool(semantic_search_ok),
        id="semantic_search_action",
        desc="[Action Node] semanticscholar.org:F1:A5 - Search and access paper details on Semantic Scholar",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P1 - Extract TLDR summary
    has_tldr = bool(target_paper.tldr_summary and target_paper.tldr_summary.strip() and len(target_paper.tldr_summary) > 10)
    evaluator.add_custom_node(
        result=bool(has_tldr),
        id="semantic_tldr_extraction",
        desc="[Perception Node] semanticscholar.org:F2:P1 - Extract TLDR summary from paper page",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P3 - Extract and compare citation statistics
    total_cites = extract_numeric(target_paper.total_citation_count)
    influential_cites = extract_numeric(target_paper.highly_influential_citation_count)
    has_citation_stats = (total_cites is not None) and (influential_cites is not None)
    evaluator.add_custom_node(
        result=bool(has_citation_stats),
        id="semantic_citation_stats",
        desc="[Perception Node] semanticscholar.org:F2:P3 - Extract total and highly influential citation counts, select highest cited paper",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F3:A8 - Switch to Citations tab
    citations_tab_ok = has_any_ci(answer, ['citations tab', 'citations page', 'citation'])
    evaluator.add_custom_node(
        result=bool(citations_tab_ok),
        id="semantic_citations_tab",
        desc="[Action Node] semanticscholar.org:F3:A8 - Switch to Citations tab for detailed citation view",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F3:A9 - Filter for Highly Influential citations
    highly_influential_filter_ok = has_any_ci(answer, ['highly influential'])
    evaluator.add_custom_node(
        result=bool(highly_influential_filter_ok),
        id="semantic_filter_influential",
        desc="[Action Node] semanticscholar.org:F3:A9 - Filter citations by 'Highly Influential' type",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F3:P2 - Identify highly influential labels
    evaluator.add_custom_node(
        result=bool(highly_influential_filter_ok),
        id="semantic_influential_label",
        desc="[Perception Node] semanticscholar.org:F3:P2 - Recognize and identify highly influential citation labels",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F3:P10 - Extract 3 subsequent works with high citations
    work1_ok = (subsequent_works.work1 and
               subsequent_works.work1.title and
               subsequent_works.work1.citation_count and
               extract_numeric(subsequent_works.work1.citation_count) is not None)
    work2_ok = (subsequent_works.work2 and
               subsequent_works.work2.title and
               subsequent_works.work2.citation_count and
               extract_numeric(subsequent_works.work2.citation_count) is not None)
    work3_ok = (subsequent_works.work3 and
               subsequent_works.work3.title and
               subsequent_works.work3.citation_count and
               extract_numeric(subsequent_works.work3.citation_count) is not None)

    subsequent_works_ok = work1_ok and work2_ok and work3_ok
    evaluator.add_custom_node(
        result=bool(subsequent_works_ok),
        id="semantic_subsequent_works",
        desc="[Perception Node] semanticscholar.org:F3:P10 - Identify 3 highly influential subsequent works with high citation counts",
        parent=semantic_scholar_node,
        critical=False
    )

    # Additional checks for subsequent works
    has_subsequent_links = (
        (subsequent_works.work1 and looks_like_url(subsequent_works.work1.semantic_scholar_link)) and
        (subsequent_works.work2 and looks_like_url(subsequent_works.work2.semantic_scholar_link)) and
        (subsequent_works.work3 and looks_like_url(subsequent_works.work3.semantic_scholar_link))
    )
    evaluator.add_custom_node(
        result=bool(has_subsequent_links),
        id="subsequent_works_links",
        desc="Semantic Scholar links provided for all 3 subsequent works",
        parent=semantic_scholar_node,
        critical=False
    )

    # 3.3 Overall completeness check
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall task completeness and output quality",
        parent=root,
        critical=False
    )

    # Check all required target paper fields are present
    target_complete = (
        has_arxiv_id and
        bool(target_paper.title) and
        has_authors and
        has_pub_date and
        has_arxiv_link and
        has_tldr and
        has_citation_stats and
        has_semantic_link
    )
    evaluator.add_custom_node(
        result=bool(target_complete),
        id="target_paper_complete",
        desc="All required target paper information fields are present",
        parent=completeness_node,
        critical=False
    )

    # Check workflow mentions key steps
    mentions_cv_category = has_any_ci(answer, ['cs.cv', 'computer vision'])
    mentions_3_papers_initial = has_any_ci(answer, ['3 papers', 'three papers'])
    mentions_comparison = has_any_ci(answer, ['highest', 'most cited', 'citation count'])

    evaluator.add_custom_node(
        result=bool(mentions_cv_category),
        id="mentions_cv_filter",
        desc="Mentions filtering for Computer Vision (cs.CV) category",
        parent=completeness_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(mentions_3_papers_initial and mentions_comparison),
        id="mentions_selection_process",
        desc="Describes the paper selection and comparison process",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
