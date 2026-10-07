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
TASK_ID = "task-7b62fb"
TASK_DESCRIPTION = 'I am a researcher in digital preservation of cultural heritage, preparing a proposal report for the museum director on AI-based artwork restoration technologies. First, search Semantic Scholar for **“AI art restoration”** or **“neural network painting restoration”**, filter for papers published since 2024, sort by citation count in descending order, and identify the 5 most-cited papers that provide a PDF download link. Record each paper’s title, authors, publication year, citation count, and paper link.\n\nThen, extract the core algorithm names from the abstracts or full texts of these 5 papers (e.g., GAN-based restoration, Transformer-based inpainting), and search GitHub for corresponding open-source implementations. Prefer projects with more than 500 stars and updates within the last year. Select the 3 most active projects and record the project name, star count, most recent update date, primary programming language, and project link. If fewer than 3 projects meet all criteria, keep the “updated within the last year” requirement, prioritize higher-star relevant projects, and clearly note which ones do not reach 500 stars. If GitHub access is restricted or requires login, document the limitation and continue by using code links from paper pages or searching via a search engine with queries like “algorithm name + GitHub” to find publicly accessible repositories.\n\nNext, search Google Patents for patents related to these algorithms. Filter for granted U.S. patents filed after 2020, identify 2–3 patents, and record the patent number, title, assignee, filing date, and patent link.\n\nFinally, visit the Getty Museum official website. Under **“What’s On”** in Current or Future exhibitions, check whether there are exhibitions or events related to digital restoration or technology-driven conservation. If found, record the exhibition name, date range, summary, and link. If none are relevant, search the Getty collections pages using the keywords **“digital”** or **“conservation”** to find cases of digitally supported restoration, and record 1–2 cases with object name, description, and link.\n\n**Output format:**  \n- 5 papers (title, authors, publication year, citation count, Semantic Scholar link)  \n- 3 GitHub projects (project name, star count, latest update date, primary programming language, GitHub link)  \n- 2–3 patents (patent number, title, assignee, filing date, Google Patents link)  \n- Getty-related exhibition(s) or collection case(s) (name, description, link)'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SemanticScholarPaper(BaseModel):
    """A single paper from Semantic Scholar"""
    title: Optional[str] = None
    authors: Optional[str] = None
    publication_year: Optional[int] = None
    citation_count: Optional[int] = None
    paper_link: Optional[str] = None
    has_pdf: Optional[bool] = None


class SemanticScholarPapers(BaseModel):
    """Collection of papers from Semantic Scholar"""
    papers: List[SemanticScholarPaper] = Field(default_factory=list)


class AlgorithmNames(BaseModel):
    """Extracted algorithm names from papers"""
    algorithm_names: List[str] = Field(default_factory=list)


class GitHubProject(BaseModel):
    """A single GitHub project"""
    project_name: Optional[str] = None
    star_count: Optional[int] = None
    latest_update_date: Optional[str] = None
    primary_language: Optional[str] = None
    project_link: Optional[str] = None
    meets_500_stars: Optional[bool] = None


class GitHubProjects(BaseModel):
    """Collection of GitHub projects"""
    projects: List[GitHubProject] = Field(default_factory=list)


class Patent(BaseModel):
    """A single patent from Google Patents"""
    patent_number: Optional[str] = None
    title: Optional[str] = None
    assignee: Optional[str] = None
    filing_date: Optional[str] = None
    patent_link: Optional[str] = None
    is_granted: Optional[bool] = None


class Patents(BaseModel):
    """Collection of patents"""
    patents: List[Patent] = Field(default_factory=list)


class GettyInfo(BaseModel):
    """Getty Museum exhibition or collection case"""
    name: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None
    date_range: Optional[str] = None
    is_exhibition: Optional[bool] = None


class GettyInfoList(BaseModel):
    """Collection of Getty information"""
    items: List[GettyInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
Extract the 5 Semantic Scholar papers from the answer. For each paper, extract:
- title: paper title
- authors: author names as stated
- publication_year: year of publication (as integer)
- citation_count: number of citations (as integer)
- paper_link: Semantic Scholar link to the paper
- has_pdf: whether the paper has PDF available (boolean)

Return all 5 papers in the papers list. If fewer than 5 are present, return what's available.
"""


def prompt_extract_algorithms() -> str:
    return """
Extract the core algorithm names mentioned in the answer that were extracted from the papers.
Examples: "GAN-based restoration", "Transformer-based inpainting", "diffusion models", etc.

Return them in the algorithm_names list.
"""


def prompt_extract_github_projects() -> str:
    return """
Extract the 3 GitHub projects from the answer. For each project, extract:
- project_name: name of the GitHub project
- star_count: number of stars (as integer)
- latest_update_date: most recent update date
- primary_language: main programming language
- project_link: GitHub link to the project
- meets_500_stars: whether the project has 500+ stars (boolean)

Return all projects in the projects list. If fewer than 3 are present, return what's available.
"""


def prompt_extract_patents() -> str:
    return """
Extract the 2-3 patents from the answer. For each patent, extract:
- patent_number: patent number
- title: patent title
- assignee: patent assignee/owner
- filing_date: filing date
- patent_link: Google Patents link
- is_granted: whether it's a granted patent (boolean)

Return all patents in the patents list.
"""


def prompt_extract_getty_info() -> str:
    return """
Extract Getty Museum exhibition(s) or collection case(s) from the answer. For each item, extract:
- name: exhibition name or object name
- description: summary or description
- link: link to the Getty page
- date_range: date range if it's an exhibition
- is_exhibition: true if it's an exhibition, false if it's a collection case

Return all items in the items list.
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


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\b(202[0-9])\b', str(text))
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def is_year_2024_or_later(year: Optional[int]) -> bool:
    if year is None:
        return False
    return year >= 2024


def is_descending_order(values: List[Optional[int]]) -> bool:
    """Check if citation counts are in descending order"""
    valid_values = [v for v in values if v is not None]
    if len(valid_values) < 2:
        return True
    for i in range(len(valid_values) - 1):
        if valid_values[i] < valid_values[i + 1]:
            return False
    return True


def is_semanticscholar_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'semanticscholar.org' in link.lower()


def is_github_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'github.com' in link.lower()


def is_google_patents_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'patents.google.com' in link.lower()


def is_getty_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return 'getty.edu' in link.lower()


def parse_date_year(date_str: Optional[str]) -> Optional[int]:
    """Extract year from date string"""
    if not date_str:
        return None
    m = re.search(r'\b(20[0-9]{2})\b', str(date_str))
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def is_within_last_year(date_str: Optional[str]) -> bool:
    """Check if update date is within last year (2024 or later)"""
    if not date_str:
        return False
    year = parse_date_year(date_str)
    if year is None:
        # Try to be lenient with relative dates
        if has_any_ci(date_str, ['months ago', 'month ago', 'weeks ago', 'week ago', 'days ago', 'day ago']):
            return True
        return False
    return year >= 2024


def is_filing_after_2020(date_str: Optional[str]) -> bool:
    """Check if filing date is after 2020"""
    if not date_str:
        return False
    year = parse_date_year(date_str)
    if year is None:
        return False
    return year > 2020


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
        prompt=prompt_extract_papers(),
        template_class=SemanticScholarPapers,
        extraction_name="semantic_scholar_papers"
    )

    algorithms_info = await evaluator.extract(
        prompt=prompt_extract_algorithms(),
        template_class=AlgorithmNames,
        extraction_name="algorithm_names"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_projects(),
        template_class=GitHubProjects,
        extraction_name="github_projects"
    )

    patents_info = await evaluator.extract(
        prompt=prompt_extract_patents(),
        template_class=Patents,
        extraction_name="patents"
    )

    getty_info = await evaluator.extract(
        prompt=prompt_extract_getty_info(),
        template_class=GettyInfoList,
        extraction_name="getty_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Semantic Scholar section
    semantic_node = evaluator.add_sequential(
        id="semantic_scholar_section",
        desc="Semantic Scholar: Search for AI art restoration papers",
        parent=root,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A3 - Date range filtering
    papers_list = papers_info.papers if papers_info and papers_info.papers else []
    years_2024_plus = [is_year_2024_or_later(p.publication_year) for p in papers_list if p.publication_year is not None]
    date_filter_ok = len(years_2024_plus) > 0 and all(years_2024_plus)

    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="semantic_date_filter",
        desc="[Action Node] semanticscholar.org:F1:A3 - Filter papers published since 2024",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Sorting by citation count
    citation_counts = [p.citation_count for p in papers_list]
    sort_ok = is_descending_order(citation_counts)

    evaluator.add_custom_node(
        result=bool(sort_ok and len(citation_counts) > 0),
        id="semantic_sort_citations",
        desc="[Action Node] semanticscholar.org:F1:A4 - Sort papers by citation count in descending order",
        parent=semantic_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F1:P2 - PDF availability recognition
    has_pdf_checks = [p.has_pdf for p in papers_list if p.has_pdf is not None]
    pdf_recognition_ok = len(has_pdf_checks) > 0 and all(has_pdf_checks)

    evaluator.add_custom_node(
        result=bool(pdf_recognition_ok),
        id="semantic_pdf_recognition",
        desc="[Perception Node] semanticscholar.org:F1:P2 - Identify papers with PDF download links",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A5 - Click into paper details
    detail_links = [is_semanticscholar_link(p.paper_link) for p in papers_list]
    detail_navigation_ok = len(detail_links) > 0 and all(detail_links)

    evaluator.add_custom_node(
        result=bool(detail_navigation_ok),
        id="semantic_detail_navigation",
        desc="[Action Node] semanticscholar.org:F1:A5 - Navigate to paper detail pages",
        parent=semantic_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P1 - Understanding TLDR/abstract content
    algorithms_extracted = algorithms_info and algorithms_info.algorithm_names and len(algorithms_info.algorithm_names) > 0
    algorithm_keywords_ok = False
    if algorithms_extracted:
        algo_text = ' '.join(algorithms_info.algorithm_names)
        algorithm_keywords_ok = has_any_ci(algo_text, ['gan', 'transformer', 'diffusion', 'neural', 'cnn', 'resnet', 'unet'])

    evaluator.add_custom_node(
        result=bool(algorithm_keywords_ok),
        id="semantic_abstract_understanding",
        desc="[Perception Node] semanticscholar.org:F2:P1 - Extract core algorithm names from abstracts/full texts",
        parent=semantic_node,
        critical=False
    )

    # Check for 5 papers
    five_papers_ok = len(papers_list) == 5
    evaluator.add_custom_node(
        result=bool(five_papers_ok),
        id="semantic_five_papers",
        desc="Collected exactly 5 most-cited papers",
        parent=semantic_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub: Search for open-source implementations of algorithms",
        parent=root,
        critical=False
    )

    projects_list = github_info.projects if github_info and github_info.projects else []

    # [Action Node] github.com:F1:A7 - Sorting by stars/activity
    star_counts = [p.star_count for p in projects_list if p.star_count is not None]
    github_sort_ok = is_descending_order(star_counts)

    evaluator.add_custom_node(
        result=bool(github_sort_ok and len(star_counts) > 0),
        id="github_sort_stars",
        desc="[Action Node] github.com:F1:A7 - Sort projects by stars/activity",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Extract project information from list
    project_info_complete = all([
        p.project_name and p.star_count is not None and p.latest_update_date and p.primary_language and p.project_link
        for p in projects_list
    ])

    evaluator.add_custom_node(
        result=bool(project_info_complete and len(projects_list) > 0),
        id="github_list_understanding",
        desc="[Perception Node] github.com:F1:P1 - Extract project name, stars, update date, language from search results",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P10 - Recognize update time (within last year)
    update_checks = [is_within_last_year(p.latest_update_date) for p in projects_list]
    update_time_ok = len(update_checks) > 0 and all(update_checks)

    evaluator.add_custom_node(
        result=bool(update_time_ok),
        id="github_update_recognition",
        desc="[Perception Node] github.com:F1:P10 - Verify projects updated within the last year",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A17 - Navigate to project page to see language/structure
    github_links = [is_github_link(p.project_link) for p in projects_list]
    github_nav_ok = len(github_links) > 0 and all(github_links)

    evaluator.add_custom_node(
        result=bool(github_nav_ok),
        id="github_project_navigation",
        desc="[Action Node] github.com:F3:A17 - Navigate to project pages to identify primary language",
        parent=github_node,
        critical=False
    )

    # Check for 3 projects
    three_projects_ok = len(projects_list) == 3
    evaluator.add_custom_node(
        result=bool(three_projects_ok),
        id="github_three_projects",
        desc="Collected 3 most active GitHub projects",
        parent=github_node,
        critical=False
    )

    # Check 500+ stars preference
    star_500_checks = [p.star_count and p.star_count >= 500 for p in projects_list]
    star_preference_ok = len(star_500_checks) > 0 and any(star_500_checks)
    evaluator.add_custom_node(
        result=bool(star_preference_ok),
        id="github_star_preference",
        desc="At least some projects meet the 500+ stars preference",
        parent=github_node,
        critical=False
    )

    # 3.3 Google Patents section
    patents_node = evaluator.add_sequential(
        id="patents_section",
        desc="Google Patents: Search for algorithm-related patents",
        parent=root,
        critical=False
    )

    patents_list = patents_info.patents if patents_info and patents_info.patents else []

    # [Action Node] patents.google.com:F1:A2 - Search with algorithm keywords
    patent_search_ok = len(patents_list) > 0 and algorithms_extracted

    evaluator.add_custom_node(
        result=bool(patent_search_ok),
        id="patents_search_keywords",
        desc="[Action Node] patents.google.com:F1:A2 - Search using extracted algorithm keywords",
        parent=patents_node,
        critical=False
    )

    # [Action Node] patents.google.com:F1:A1 - Multi-condition filtering (date, status, country)
    filing_date_checks = [is_filing_after_2020(p.filing_date) for p in patents_list]
    granted_checks = [p.is_granted for p in patents_list if p.is_granted is not None]

    filter_ok = (len(filing_date_checks) > 0 and all(filing_date_checks) and
                 len(granted_checks) > 0 and all(granted_checks))

    evaluator.add_custom_node(
        result=bool(filter_ok),
        id="patents_filtering",
        desc="[Action Node] patents.google.com:F1:A1 - Filter for granted U.S. patents filed after 2020",
        parent=patents_node,
        critical=False
    )

    # [Perception Node] patents.google.com:F3:P1 - Extract patent metadata
    patent_info_complete = all([
        p.patent_number and p.title and p.assignee and p.filing_date and p.patent_link
        for p in patents_list
    ])

    evaluator.add_custom_node(
        result=bool(patent_info_complete and len(patents_list) > 0),
        id="patents_metadata_extraction",
        desc="[Perception Node] patents.google.com:F3:P1 - Extract patent number, title, assignee, filing date",
        parent=patents_node,
        critical=False
    )

    # [Perception Node] patents.google.com:F3:P2 - Recognize legal status (granted)
    legal_status_ok = len(granted_checks) > 0 and all(granted_checks)

    evaluator.add_custom_node(
        result=bool(legal_status_ok),
        id="patents_legal_status",
        desc="[Perception Node] patents.google.com:F3:P2 - Verify granted patent status",
        parent=patents_node,
        critical=False
    )

    # Check for 2-3 patents
    patents_count_ok = 2 <= len(patents_list) <= 3
    evaluator.add_custom_node(
        result=bool(patents_count_ok),
        id="patents_count",
        desc="Collected 2-3 patents as requested",
        parent=patents_node,
        critical=False
    )

    # 3.4 Getty Museum section
    getty_node = evaluator.add_sequential(
        id="getty_section",
        desc="Getty Museum: Check exhibitions or collection cases",
        parent=root,
        critical=False
    )

    getty_items = getty_info.items if getty_info and getty_info.items else []

    # [Action Node] getty.edu:F2:A1 - Navigate tabs (Current/Future exhibitions)
    has_getty_content = len(getty_items) > 0
    mentions_whats_on = has_any_ci(answer, ["what's on", "whats on", "current", "future", "exhibition"])

    evaluator.add_custom_node(
        result=bool(has_getty_content and mentions_whats_on),
        id="getty_navigation_tabs",
        desc="[Action Node] getty.edu:F2:A1 - Navigate to 'What's On' and check Current/Future exhibitions",
        parent=getty_node,
        critical=False
    )

    # [Action Node] getty.edu:F2:A7 - Click into exhibition/collection details
    getty_links = [is_getty_link(item.link) for item in getty_items if item.link]
    getty_detail_ok = len(getty_links) > 0 and all(getty_links)

    evaluator.add_custom_node(
        result=bool(getty_detail_ok),
        id="getty_detail_navigation",
        desc="[Action Node] getty.edu:F2:A7 - Navigate to exhibition/collection detail pages",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F1:P1 - Recognize objects/images (fallback scenario)
    has_collection_cases = any([not item.is_exhibition for item in getty_items if item.is_exhibition is not None])
    collection_keywords = has_any_ci(answer, ["collection", "digital", "conservation", "object"])

    evaluator.add_custom_node(
        result=bool((has_collection_cases and collection_keywords) or (has_getty_content and not has_collection_cases)),
        id="getty_content_recognition",
        desc="[Perception Node] getty.edu:F1:P1 - Identify exhibitions or collection objects with digital/conservation context",
        parent=getty_node,
        critical=False
    )

    # Check Getty information completeness
    getty_info_complete = all([
        item.name and item.description and item.link
        for item in getty_items
    ])

    evaluator.add_custom_node(
        result=bool(getty_info_complete and len(getty_items) > 0),
        id="getty_info_complete",
        desc="Extracted complete Getty exhibition or collection information",
        parent=getty_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
