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
TASK_ID = "task-a8a73c"
TASK_DESCRIPTION = 'I am currently preparing a presentation for a course on ancient Greek art and would like to compile a systematic collection of study materials.\n\nFirst, go to the Getty Museum website and find 5 ancient Greek artworks from the 5th-4th century BCE (Classical period). These artworks must have Open Content high-resolution images available for download. For each artwork, record its name, date, material, and accession number.\n\nNext, using the information from these 5 artworks, search Google Scholar for relevant academic papers. Find 3 English-language papers that have been cited more than 50 times and published after 2015. These papers should discuss either the artistic style of the Classical Greek period or the types of artifacts (e.g., pottery, sculpture) represented by the selected artworks.\n\nFinally, go to Our World in Data and find charts related to population or urbanization during the ancient Greek civilization period (800-300 BCE). Use the timeline feature to observe development trends during this era, which will help me understand the socio-economic background behind the artistic prosperity.\n\n**Output:**\n\n*   For each Getty artwork: Name, Date, Material, Accession Number, and the link to its Getty detail page.\n*   For the 3 academic papers: Title, Author(s), Journal, Publication Year, Citation Count, and the Google Scholar link.\n*   For Our World in Data: The title of the relevant chart(s), a data summary (e.g., peak year and value), and the chart link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GettyArtwork(BaseModel):
    """Single Getty artwork extracted from the answer"""
    name: Optional[str] = None
    date: Optional[str] = None
    material: Optional[str] = None
    accession_number: Optional[str] = None
    link: Optional[str] = None


class GettyArtworks(BaseModel):
    """Collection of Getty artworks extracted from the answer"""
    artworks: List[GettyArtwork] = Field(default_factory=list)


class ScholarPaper(BaseModel):
    """Single Google Scholar paper extracted from the answer"""
    title: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    citations: Optional[int] = None
    link: Optional[str] = None


class ScholarPapers(BaseModel):
    """Collection of Google Scholar papers extracted from the answer"""
    papers: List[ScholarPaper] = Field(default_factory=list)


class OWIDChart(BaseModel):
    """Our World in Data chart information extracted from the answer"""
    title: Optional[str] = None
    peak_year: Optional[str] = None
    peak_value: Optional[str] = None
    link: Optional[str] = None
    trend_description: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_getty_artworks() -> str:
    return """
Extract all Getty Museum artworks mentioned in the answer. For each artwork, extract:
- name: the artwork's name or title
- date: the date or date range (should be 5th-4th century BCE or Classical period)
- material: the material (e.g., marble, bronze, terracotta, ceramic)
- accession_number: the Getty accession number
- link: the URL to the Getty detail page

Return up to 5 artworks. If any field is missing for an artwork, set it to null.
"""


def prompt_extract_scholar_papers() -> str:
    return """
Extract all Google Scholar papers mentioned in the answer. For each paper, extract:
- title: the paper title
- authors: the author names
- journal: the journal or publication venue
- year: the publication year as an integer
- citations: the citation count as an integer
- link: the Google Scholar link

Return up to 3 papers. If any field is missing, set it to null.
"""


def prompt_extract_owid_chart() -> str:
    return """
Extract the Our World in Data chart information from the answer:
- title: the chart title
- peak_year: the year with peak value during 800-300 BCE period
- peak_value: the peak value (include units if present)
- link: the chart URL
- trend_description: any description of the trend during 800-300 BCE

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


def looks_like_classical_period(date_text: Optional[str]) -> bool:
    if not date_text:
        return False
    # Check for BCE/BC and century markers or specific year ranges
    has_bce = has_any_ci(date_text, ['bce', 'b.c.e', 'bc', 'b.c'])
    has_century = has_any_ci(date_text, ['5th', '4th', 'fifth', 'fourth', 'century', 'classical'])
    # Check for year ranges like 500-400, 450-350, etc.
    year_pattern = re.search(r'\b[4-5]\d{2}\b', date_text)
    return (has_bce and has_century) or (has_bce and year_pattern is not None)


def looks_like_material(material_text: Optional[str]) -> bool:
    if not material_text:
        return False
    common_materials = ['marble', 'bronze', 'terracotta', 'ceramic', 'stone', 'clay',
                       'limestone', 'pottery', 'gold', 'silver', 'wood', 'ivory']
    return has_any_ci(material_text, common_materials)


def looks_like_accession_number(acc_text: Optional[str]) -> bool:
    if not acc_text:
        return False
    # Getty accession numbers typically contain periods and numbers
    return bool(re.search(r'\d+\.\w+', acc_text))


def looks_like_getty_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'getty.edu' in link.lower()


def looks_like_scholar_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'scholar.google' in link.lower()


def looks_like_owid_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'ourworldindata.org' in link.lower()


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\d+', str(text))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
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
    getty_data = await evaluator.extract(
        prompt=prompt_extract_getty_artworks(),
        template_class=GettyArtworks,
        extraction_name="getty_artworks"
    )

    scholar_data = await evaluator.extract(
        prompt=prompt_extract_scholar_papers(),
        template_class=ScholarPapers,
        extraction_name="scholar_papers"
    )

    owid_data = await evaluator.extract(
        prompt=prompt_extract_owid_chart(),
        template_class=OWIDChart,
        extraction_name="owid_chart"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Getty Museum section
    getty_node = evaluator.add_sequential(
        id="getty_section",
        desc="Getty Museum artworks from Classical period (5th-4th century BCE)",
        parent=root,
        critical=False
    )

    # [Action Node] getty.edu:F1:A2 - Navigate to Getty Museum collection
    getty_nav_ok = has_any_ci(answer, ['getty']) and (
        has_any_ci(answer, ['museum', 'collection', 'art', 'explore'])
    )
    evaluator.add_custom_node(
        result=bool(getty_nav_ok),
        id="getty_navigation",
        desc="[Action Node] getty.edu:F1:A2 - Navigate to Getty Museum website and access the art collection",
        parent=getty_node,
        critical=False
    )

    # Check if we have 5 artworks
    artworks = getty_data.artworks if getty_data else []
    has_five_artworks = len(artworks) >= 5

    evaluator.add_custom_node(
        result=bool(has_five_artworks),
        id="getty_five_artworks",
        desc="Found 5 ancient Greek artworks from the Classical period",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F1:P1 - Identify Greek artworks visually
    greek_keywords = has_any_ci(answer, ['greek', 'greece', 'hellenic', 'attic'])
    evaluator.add_custom_node(
        result=bool(greek_keywords and len(artworks) > 0),
        id="getty_visual_identification",
        desc="[Perception Node] getty.edu:F1:P1 - Visually identify ancient Greek artworks",
        parent=getty_node,
        critical=False
    )

    # Check dates are in Classical period
    dates_ok = sum(1 for a in artworks if looks_like_classical_period(a.date)) >= min(3, len(artworks))
    evaluator.add_custom_node(
        result=bool(dates_ok),
        id="getty_classical_dates",
        desc="[Perception Node] getty.edu:F1:P1 - Verify artworks are from 5th-4th century BCE",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F1:P3 - Extract material information
    materials_ok = sum(1 for a in artworks if looks_like_material(a.material)) >= min(3, len(artworks))
    evaluator.add_custom_node(
        result=bool(materials_ok),
        id="getty_material_extraction",
        desc="[Perception Node] getty.edu:F1:P3 - Extract material information from artworks",
        parent=getty_node,
        critical=False
    )

    # Check accession numbers
    accession_ok = sum(1 for a in artworks if looks_like_accession_number(a.accession_number)) >= min(3, len(artworks))
    evaluator.add_custom_node(
        result=bool(accession_ok),
        id="getty_accession_numbers",
        desc="Extracted valid accession numbers for artworks",
        parent=getty_node,
        critical=False
    )

    # Check Open Content mentioned
    open_content_ok = has_any_ci(answer, ['open content', 'high-resolution', 'download'])
    evaluator.add_custom_node(
        result=bool(open_content_ok),
        id="getty_open_content",
        desc="Verified artworks have Open Content high-resolution images available",
        parent=getty_node,
        critical=False
    )

    # Check Getty links provided
    links_ok = sum(1 for a in artworks if looks_like_getty_link(a.link)) >= min(3, len(artworks))
    evaluator.add_custom_node(
        result=bool(links_ok),
        id="getty_links",
        desc="Provided Getty detail page links for artworks",
        parent=getty_node,
        critical=False
    )

    # 3.2 Google Scholar section
    scholar_node = evaluator.add_sequential(
        id="scholar_section",
        desc="Google Scholar academic papers on Classical Greek art",
        parent=root,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A1 - Search Google Scholar
    scholar_search_ok = has_any_ci(answer, ['google scholar', 'scholar']) and (
        has_any_ci(answer, ['classical', 'greek', 'art', 'pottery', 'sculpture'])
    )
    evaluator.add_custom_node(
        result=bool(scholar_search_ok),
        id="scholar_search",
        desc="[Action Node] scholar.google.com:F1:A1 - Search Google Scholar for papers on Classical Greek art",
        parent=scholar_node,
        critical=False
    )

    # Check if we have 3 papers
    papers = scholar_data.papers if scholar_data else []
    has_three_papers = len(papers) >= 3

    evaluator.add_custom_node(
        result=bool(has_three_papers),
        id="scholar_three_papers",
        desc="Found 3 relevant academic papers",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P1 - Understand paper topics
    topics_relevant = sum(1 for p in papers if p.title and has_any_ci(p.title,
        ['greek', 'classical', 'art', 'pottery', 'sculpture', 'ceramic', 'hellenic', 'attic'])) >= min(2, len(papers))
    evaluator.add_custom_node(
        result=bool(topics_relevant),
        id="scholar_topic_understanding",
        desc="[Perception Node] scholar.google.com:F1:P1 - Verify papers discuss Classical Greek art or artifacts",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A8 - Click paper details
    has_details = sum(1 for p in papers if p.authors and p.journal) >= min(2, len(papers))
    evaluator.add_custom_node(
        result=bool(has_details),
        id="scholar_paper_details",
        desc="[Action Node] scholar.google.com:F1:A8 - Access paper details to extract authors and journal",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A6 - Filter by year (after 2015)
    years_after_2015 = sum(1 for p in papers if p.year and p.year >= 2015) >= min(2, len(papers))
    evaluator.add_custom_node(
        result=bool(years_after_2015),
        id="scholar_year_filter",
        desc="[Action Node] scholar.google.com:F3:A6 - Filter papers published after 2015",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F4:P3 - Identify citation counts
    citations_over_50 = sum(1 for p in papers if p.citations and p.citations > 50) >= min(2, len(papers))
    evaluator.add_custom_node(
        result=bool(citations_over_50),
        id="scholar_citation_identification",
        desc="[Perception Node] scholar.google.com:F4:P3 - Identify papers with over 50 citations",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F4:A9 - View citation counts
    has_citation_data = sum(1 for p in papers if p.citations is not None) >= min(2, len(papers))
    evaluator.add_custom_node(
        result=bool(has_citation_data),
        id="scholar_view_citations",
        desc="[Action Node] scholar.google.com:F4:A9 - View and extract citation counts",
        parent=scholar_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A15 - Browse multiple pages
    browse_mention = has_any_ci(answer, ['page', 'browse', 'search results', 'multiple'])
    evaluator.add_custom_node(
        result=bool(browse_mention or len(papers) >= 3),
        id="scholar_browse_pages",
        desc="[Action Node] scholar.google.com:F3:A15 - Browse through search results to find qualifying papers",
        parent=scholar_node,
        critical=False
    )

    # Check Scholar links provided
    scholar_links_ok = sum(1 for p in papers if looks_like_scholar_link(p.link)) >= min(2, len(papers))
    evaluator.add_custom_node(
        result=bool(scholar_links_ok),
        id="scholar_links",
        desc="Provided Google Scholar links for papers",
        parent=scholar_node,
        critical=False
    )

    # 3.3 Our World in Data section
    owid_node = evaluator.add_sequential(
        id="owid_section",
        desc="Our World in Data charts on ancient Greek population/urbanization",
        parent=root,
        critical=False
    )

    # Check OWID mentioned
    owid_mentioned = has_any_ci(answer, ['our world in data', 'ourworldindata'])
    evaluator.add_custom_node(
        result=bool(owid_mentioned),
        id="owid_navigation",
        desc="Navigate to Our World in Data website",
        parent=owid_node,
        critical=False
    )

    # Check chart found
    has_chart = bool(owid_data and owid_data.title)
    evaluator.add_custom_node(
        result=bool(has_chart),
        id="owid_chart_found",
        desc="Found chart related to ancient Greek civilization",
        parent=owid_node,
        critical=False
    )

    # Check relevant topic (population or urbanization)
    relevant_topic = owid_data and owid_data.title and has_any_ci(owid_data.title,
        ['population', 'urban', 'city', 'cities', 'demographic'])
    evaluator.add_custom_node(
        result=bool(relevant_topic),
        id="owid_relevant_topic",
        desc="Chart covers population or urbanization data",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F2:A1 - Use timeline feature
    timeline_used = has_any_ci(answer, ['timeline', 'time', 'slider', '800', '300 bce', 'period'])
    evaluator.add_custom_node(
        result=bool(timeline_used),
        id="owid_timeline_usage",
        desc="[Action Node] ourworldindata.org:F2:A1 - Use timeline feature to view 800-300 BCE period",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A8 - Hover to view data values
    has_specific_values = bool(owid_data and (owid_data.peak_value or owid_data.peak_year))
    evaluator.add_custom_node(
        result=bool(has_specific_values),
        id="owid_hover_values",
        desc="[Action Node] ourworldindata.org:F1:A8 - Hover over chart to extract specific data values",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P9 - Read precise values
    has_precise_data = bool(owid_data and owid_data.peak_value and contains_digits(owid_data.peak_value))
    evaluator.add_custom_node(
        result=bool(has_precise_data),
        id="owid_precise_values",
        desc="[Perception Node] ourworldindata.org:F1:P9 - Extract precise numerical values from chart",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P3 - Identify peak values
    has_peak_info = bool(owid_data and owid_data.peak_year)
    evaluator.add_custom_node(
        result=bool(has_peak_info),
        id="owid_peak_identification",
        desc="[Perception Node] ourworldindata.org:F1:P3 - Identify peak year and value during 800-300 BCE",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P1 - Interpret trend
    has_trend = bool(owid_data and owid_data.trend_description) or has_any_ci(answer,
        ['trend', 'increase', 'decrease', 'growth', 'decline', 'development'])
    evaluator.add_custom_node(
        result=bool(has_trend),
        id="owid_trend_interpretation",
        desc="[Perception Node] ourworldindata.org:F1:P1 - Interpret development trends during 800-300 BCE",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F2:P3 - Identify historical extremes
    historical_context = has_any_ci(answer, ['ancient', 'greek', 'civilization', 'classical', 'bce'])
    evaluator.add_custom_node(
        result=bool(historical_context and has_peak_info),
        id="owid_historical_extremes",
        desc="[Perception Node] ourworldindata.org:F2:P3 - Identify historical extremes during ancient Greek period",
        parent=owid_node,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A4 - Switch between view types
    view_mention = has_any_ci(answer, ['chart', 'map', 'table', 'view', 'visualization'])
    evaluator.add_custom_node(
        result=bool(view_mention),
        id="owid_view_switching",
        desc="[Action Node] ourworldindata.org:F1:A4 - Navigate chart views to find appropriate data",
        parent=owid_node,
        critical=False
    )

    # Check OWID link provided
    owid_link_ok = bool(owid_data and looks_like_owid_link(owid_data.link))
    evaluator.add_custom_node(
        result=bool(owid_link_ok),
        id="owid_link",
        desc="Provided Our World in Data chart link",
        parent=owid_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
