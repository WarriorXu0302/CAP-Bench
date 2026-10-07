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
TASK_ID = "task-b48b7d"
TASK_DESCRIPTION = 'I’m writing the literature review for my master’s thesis on “the digital divide and educational inequality,” and I need to trace the scholarly evolution of the core concepts.\n\nFirst, go to Semantic Scholar and search for “digital divide education inequality.” Try to find 3 high-impact papers published since 2020 with more than 100 citations. If only 1–2 papers meet the criteria, keep the actual number found and explicitly note the final count in the results.\n\nFor the first paper, open its detail page and check the citation analysis. Identify 2 Highly Influential Citations as upstream papers in the core citation chain. If fewer than 2 are available, supplement with regular citations and clearly label which are highly influential and which are regular.\n\nThen go to Our World in Data and search for data related to “digital divide” and “education inequality.” Filter for education or technology themes, find 1 chart showing global or regional data, and review the data source notes.\n\nFinally, go to Open Library and search for the classic book “The Digital Divide.” Find the first edition or the most influential edition, then check its detailed publication information and subject tags.\n\nOutput:\n- For each Semantic Scholar paper: title, authors, publication year, citation count, TLDR summary, paper link  \n- For the cited papers: title, citation type (background/method/results), paper link  \n- For the Our World in Data chart: chart title, data time range, summary of data source notes, chart link  \n- For the Open Library book: title, first publication year, publisher, page count, ISBN, list of subject tags, book link'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SemanticScholarPaper(BaseModel):
    """Single paper details from Semantic Scholar"""
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    citations: Optional[int] = None
    tldr: Optional[str] = None
    link: Optional[str] = None


class SemanticScholarPapers(BaseModel):
    """Collection of papers from Semantic Scholar search"""
    papers: List[SemanticScholarPaper] = Field(default_factory=list)
    count_mentioned: Optional[int] = None


class CitedPaper(BaseModel):
    """Cited paper details"""
    title: Optional[str] = None
    citation_type: Optional[str] = None
    is_highly_influential: Optional[bool] = None
    link: Optional[str] = None


class CitedPapers(BaseModel):
    """Collection of cited papers"""
    papers: List[CitedPaper] = Field(default_factory=list)


class OurWorldInDataChart(BaseModel):
    """Chart details from Our World in Data"""
    title: Optional[str] = None
    time_range: Optional[str] = None
    source_notes: Optional[str] = None
    link: Optional[str] = None


class OpenLibraryBook(BaseModel):
    """Book details from Open Library"""
    title: Optional[str] = None
    first_publication_year: Optional[int] = None
    publisher: Optional[str] = None
    page_count: Optional[int] = None
    isbn: Optional[str] = None
    subjects: List[str] = Field(default_factory=list)
    link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_semantic_scholar_papers() -> str:
    return """
Extract the Semantic Scholar papers found in the answer for "digital divide education inequality" published since 2020 with more than 100 citations.

Return:
- papers: list of papers with title, authors, year, citations, tldr, link
- count_mentioned: the number of papers the answer explicitly states it found (e.g., "found 3 papers" or "found 2 papers")

If any field is missing, set it to null or empty list.
"""


def prompt_extract_cited_papers() -> str:
    return """
Extract the cited papers (Highly Influential Citations) from the first Semantic Scholar paper's citation analysis.

Return:
- papers: list with title, citation_type (background/method/results), is_highly_influential (true/false), link

If any field is missing, set it to null or false.
"""


def prompt_extract_owid_chart() -> str:
    return """
Extract the Our World in Data chart details about digital divide and education inequality.

Return:
- title: chart title
- time_range: data time range
- source_notes: summary of data source notes
- link: chart link

If any field is missing, set it to null.
"""


def prompt_extract_openlibrary_book() -> str:
    return """
Extract the Open Library book details for "The Digital Divide."

Return:
- title: book title
- first_publication_year: first publication year
- publisher: publisher name
- page_count: number of pages
- isbn: ISBN number
- subjects: list of subject tags
- link: book link

If any field is missing, set it to null or empty list.
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


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\b(20\d{2})\b', str(text))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


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
    papers_info = await evaluator.extract(
        prompt=prompt_extract_semantic_scholar_papers(),
        template_class=SemanticScholarPapers,
        extraction_name="semantic_scholar_papers"
    )

    cited_info = await evaluator.extract(
        prompt=prompt_extract_cited_papers(),
        template_class=CitedPapers,
        extraction_name="cited_papers"
    )

    owid_info = await evaluator.extract(
        prompt=prompt_extract_owid_chart(),
        template_class=OurWorldInDataChart,
        extraction_name="owid_chart"
    )

    book_info = await evaluator.extract(
        prompt=prompt_extract_openlibrary_book(),
        template_class=OpenLibraryBook,
        extraction_name="openlibrary_book"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Semantic Scholar section
    semanticscholar_node = evaluator.add_sequential(
        id="semanticscholar_section",
        desc="Semantic Scholar paper search and citation analysis",
        parent=root,
        critical=False
    )

    # Check if Semantic Scholar is mentioned
    ss_mentioned = has_any_ci(answer, ['semantic scholar', 'semanticscholar'])

    # [Action Node] semanticscholar.org:F1:A3 - Date range filtering
    papers_since_2020 = all(
        (p.year is None or p.year >= 2020)
        for p in (papers_info.papers if papers_info and papers_info.papers else [])
    )
    date_filter_ok = ss_mentioned and papers_since_2020 and len(papers_info.papers if papers_info else []) > 0

    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="semanticscholar_date_filter",
        desc="[Action Node] semanticscholar.org:F1:A3 - Filter papers published since 2020",
        parent=semanticscholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Sort by citations
    papers_over_100 = all(
        (p.citations is None or p.citations > 100)
        for p in (papers_info.papers if papers_info and papers_info.papers else [])
    )
    citation_sort_ok = ss_mentioned and papers_over_100 and len(papers_info.papers if papers_info else []) > 0

    evaluator.add_custom_node(
        result=bool(citation_sort_ok),
        id="semanticscholar_citation_sort",
        desc="[Action Node] semanticscholar.org:F1:A4 - Sort by citation count to find high-impact papers (>100 citations)",
        parent=semanticscholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A5 - Click to enter details
    detail_action_ok = ss_mentioned and (
        has_any_ci(answer, ['detail', 'citation analysis', 'citations']) or
        (cited_info and cited_info.papers and len(cited_info.papers) > 0)
    )

    evaluator.add_custom_node(
        result=bool(detail_action_ok),
        id="semanticscholar_enter_details",
        desc="[Action Node] semanticscholar.org:F1:A5 - Click on first paper to enter detail page",
        parent=semanticscholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F1:P1 - TLDR summary identification
    has_tldr = any(
        (p.tldr and len(p.tldr.strip()) > 0)
        for p in (papers_info.papers if papers_info and papers_info.papers else [])
    )

    evaluator.add_custom_node(
        result=bool(has_tldr),
        id="semanticscholar_tldr_recognition",
        desc="[Perception Node] semanticscholar.org:F1:P1 - Identify TLDR summaries for papers",
        parent=semanticscholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F1:P2 - High-impact label recognition
    high_impact_recognition = papers_over_100 and len(papers_info.papers if papers_info else []) > 0

    evaluator.add_custom_node(
        result=bool(high_impact_recognition),
        id="semanticscholar_high_impact_label",
        desc="[Perception Node] semanticscholar.org:F1:P2 - Recognize high-impact papers based on citation count",
        parent=semanticscholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F3:A8 - Citations tab switch
    citations_tab_ok = has_any_ci(answer, ['citation', 'highly influential']) or (
        cited_info and cited_info.papers and len(cited_info.papers) > 0
    )

    evaluator.add_custom_node(
        result=bool(citations_tab_ok),
        id="semanticscholar_citations_tab",
        desc="[Action Node] semanticscholar.org:F3:A8 - Switch to Citations tab to view citing papers",
        parent=semanticscholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F3:A9 - Citation type filtering
    has_highly_influential = any(
        (p.is_highly_influential == True)
        for p in (cited_info.papers if cited_info and cited_info.papers else [])
    )
    citation_filter_ok = has_highly_influential or has_any_ci(answer, ['highly influential'])

    evaluator.add_custom_node(
        result=bool(citation_filter_ok),
        id="semanticscholar_citation_type_filter",
        desc="[Action Node] semanticscholar.org:F3:A9 - Filter for Highly Influential Citations",
        parent=semanticscholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F3:P3 - Citation statistics understanding
    citation_stats_ok = (cited_info and cited_info.papers and len(cited_info.papers) > 0) and (
        has_highly_influential or has_any_ci(answer, ['highly influential', 'regular citation'])
    )

    evaluator.add_custom_node(
        result=bool(citation_stats_ok),
        id="semanticscholar_citation_stats",
        desc="[Perception Node] semanticscholar.org:F3:P3 - Understand citation statistics and highly influential designation",
        parent=semanticscholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F3:P10 - Citation type label recognition
    has_citation_types = any(
        (p.citation_type and len(p.citation_type.strip()) > 0)
        for p in (cited_info.papers if cited_info and cited_info.papers else [])
    )

    evaluator.add_custom_node(
        result=bool(has_citation_types),
        id="semanticscholar_citation_type_labels",
        desc="[Perception Node] semanticscholar.org:F3:P10 - Identify citation type labels (background/method/results)",
        parent=semanticscholar_node,
        critical=False
    )

    # 3.2 Our World in Data section
    owid_node = evaluator.add_sequential(
        id="owid_section",
        desc="Our World in Data chart search and data source review",
        parent=root,
        critical=False
    )

    owid_mentioned = has_any_ci(answer, ['our world in data', 'ourworldindata'])

    # [Action Node] ourworldindata.org:F9:A13 - Topic navigation
    topic_navigation_ok = owid_mentioned and (
        has_any_ci(answer, ['education', 'technology', 'digital']) or
        (owid_info and owid_info.title and has_any_ci(owid_info.title, ['education', 'technology', 'digital']))
    )

    evaluator.add_custom_node(
        result=bool(topic_navigation_ok),
        id="owid_topic_navigation",
        desc="[Action Node] ourworldindata.org:F9:A13 - Navigate using topic filters for education or technology",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F9:A14 - Multi-condition filtering
    multi_filter_ok = topic_navigation_ok

    evaluator.add_custom_node(
        result=bool(multi_filter_ok),
        id="owid_multi_filter",
        desc="[Action Node] ourworldindata.org:F9:A14 - Apply topic category filters for education or technology themes",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F11:A12 - Expand data notes
    data_notes_ok = owid_info and owid_info.source_notes and len(owid_info.source_notes.strip()) > 10

    evaluator.add_custom_node(
        result=bool(data_notes_ok),
        id="owid_expand_data_notes",
        desc="[Action Node] ourworldindata.org:F11:A12 - Expand collapsed data source notes panel",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F11:P13 - Collapsed panel state recognition
    panel_state_ok = data_notes_ok or has_any_ci(answer, ['source', 'data source', 'methodology'])

    evaluator.add_custom_node(
        result=bool(panel_state_ok),
        id="owid_panel_state_recognition",
        desc="[Perception Node] ourworldindata.org:F11:P13 - Recognize and interact with collapsed data notes panel",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A4 - Chart view switching
    chart_view_ok = owid_info and owid_info.title and (
        has_any_ci(answer, ['chart', 'map', 'global', 'regional']) or
        owid_info.time_range
    )

    evaluator.add_custom_node(
        result=bool(chart_view_ok),
        id="owid_chart_view_switch",
        desc="[Action Node] ourworldindata.org:F1:A4 - Switch between Chart/Map/Table views to verify data display",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P1 - Chart trend judgment
    trend_ok = owid_info and owid_info.time_range and len(owid_info.time_range.strip()) > 0

    evaluator.add_custom_node(
        result=bool(trend_ok),
        id="owid_chart_trend",
        desc="[Perception Node] ourworldindata.org:F1:P1 - Understand chart data trends and time ranges",
        parent=owid_node,
        critical=False
    )

    # 3.3 Open Library section
    openlibrary_node = evaluator.add_sequential(
        id="openlibrary_section",
        desc="Open Library book search and detailed information",
        parent=root,
        critical=False
    )

    ol_mentioned = has_any_ci(answer, ['open library', 'openlibrary'])

    # [Action Node] openlibrary.org:F1:A1 - Search scope specification
    search_scope_ok = ol_mentioned and book_info and book_info.title and has_any_ci(book_info.title, ['digital divide'])

    evaluator.add_custom_node(
        result=bool(search_scope_ok),
        id="openlibrary_search_scope",
        desc="[Action Node] openlibrary.org:F1:A1 - Specify search type (Title/All) for 'The Digital Divide'",
        parent=openlibrary_node,
        critical=False
    )

    # [Action Node] openlibrary.org:F1:A4 - Sort selection
    sort_selection_ok = book_info and book_info.first_publication_year and (
        has_any_ci(answer, ['first edition', 'first published', 'influential edition']) or
        book_info.first_publication_year
    )

    evaluator.add_custom_node(
        result=bool(sort_selection_ok),
        id="openlibrary_sort_selection",
        desc="[Action Node] openlibrary.org:F1:A4 - Sort by First Published or Most Editions to find influential edition",
        parent=openlibrary_node,
        critical=False
    )

    # [Action Node] openlibrary.org:F1:A6 - Click to enter book details
    book_detail_ok = book_info and (
        book_info.publisher or book_info.page_count or book_info.isbn or
        (book_info.subjects and len(book_info.subjects) > 0)
    )

    evaluator.add_custom_node(
        result=bool(book_detail_ok),
        id="openlibrary_enter_details",
        desc="[Action Node] openlibrary.org:F1:A6 - Click on book card to enter detail page",
        parent=openlibrary_node,
        critical=False
    )

    # [Action Node] openlibrary.org:F3:A7 - Details tab switch
    details_tab_ok = book_detail_ok and (
        book_info.publisher or book_info.page_count or book_info.isbn
    )

    evaluator.add_custom_node(
        result=bool(details_tab_ok),
        id="openlibrary_details_tab",
        desc="[Action Node] openlibrary.org:F3:A7 - Switch to Details tab for publication information",
        parent=openlibrary_node,
        critical=False
    )

    # [Action Node] openlibrary.org:F3:A8 - Expand detailed metadata
    metadata_expanded_ok = book_info and (
        book_info.page_count or book_info.isbn or book_info.publisher
    )

    evaluator.add_custom_node(
        result=bool(metadata_expanded_ok),
        id="openlibrary_expand_metadata",
        desc="[Action Node] openlibrary.org:F3:A8 - Expand publication information panel in Details tab",
        parent=openlibrary_node,
        critical=False
    )

    # [Perception Node] openlibrary.org:F3:P2 - Publication metadata understanding
    pub_metadata_ok = book_info and book_info.publisher and book_info.page_count and book_info.isbn

    evaluator.add_custom_node(
        result=bool(pub_metadata_ok),
        id="openlibrary_pub_metadata",
        desc="[Perception Node] openlibrary.org:F3:P2 - Identify and extract publication metadata (publisher, pages, ISBN)",
        parent=openlibrary_node,
        critical=False
    )

    # [Perception Node] openlibrary.org:F3:P4 - Subject tag recognition
    subjects_ok = book_info and book_info.subjects and len(book_info.subjects) > 0

    evaluator.add_custom_node(
        result=bool(subjects_ok),
        id="openlibrary_subject_tags",
        desc="[Perception Node] openlibrary.org:F3:P4 - Identify and extract subject classification tags",
        parent=openlibrary_node,
        critical=False
    )

    # [Perception Node] openlibrary.org:F3:P9 - Edition information recognition
    edition_info_ok = book_info and book_info.first_publication_year and (
        has_any_ci(answer, ['first edition', 'edition', 'influential'])
    )

    evaluator.add_custom_node(
        result=bool(edition_info_ok),
        id="openlibrary_edition_recognition",
        desc="[Perception Node] openlibrary.org:F3:P9 - Recognize and compare edition information to find first or influential edition",
        parent=openlibrary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
