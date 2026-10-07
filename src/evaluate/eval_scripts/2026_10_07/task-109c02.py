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
TASK_ID = "task-109c02"
TASK_DESCRIPTION = "As a first-year graduate student, my advisor has tasked me with researching the 'LLM Watermarking' field, but I'm concerned about running into pitfalls. Please help me find a paper on Google Scholar published after 2023 with a relatively high citation count (at least 50 citations). Crucially, I need one that comes with runnable code. After finding a suitable paper, please navigate to its detail page and look for a GitHub link (typically located next to the PDF link or within the description). If one exists, click on the GitHub repository and briefly review the 'Open Issues' in its issue list. Specifically, look for unaddressed error reports such as 'Installation failed' or 'Environment error'. Please select the repository that appears to be the best maintained and has the fewest complaints. Then, provide me with its GitHub URL, the paper's title, and the current number of open issues."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Paper information extracted from the answer"""
    paper_title: Optional[str] = None
    publication_year: Optional[str] = None
    citation_count_text: Optional[str] = None
    github_url: Optional[str] = None


class IssueInfo(BaseModel):
    """GitHub issue information extracted from the answer"""
    open_issues_count_text: Optional[str] = None
    mentions_error_reports: Optional[bool] = None
    mentions_maintenance_quality: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
Extract the paper information from the answer about the LLM Watermarking research paper found on Google Scholar.

Return:
- paper_title: the exact title of the paper as stated in the answer.
- publication_year: the year the paper was published (e.g., "2023", "2024").
- citation_count_text: the citation count exactly as mentioned (include number and any units/context).
- github_url: the GitHub repository URL provided in the answer.

If any field is missing, set it to null.
"""


def prompt_extract_issue_info() -> str:
    return """
Extract information about the GitHub repository's open issues from the answer.

Return:
- open_issues_count_text: the number of open issues exactly as stated (include units if present).
- mentions_error_reports: true if the answer discusses checking for error reports like 'Installation failed' or 'Environment error', false otherwise.
- mentions_maintenance_quality: true if the answer discusses the repository's maintenance quality or complaint levels, false otherwise.

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


def extract_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'(\d+)', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def looks_like_year_after_2023(text: Optional[str]) -> bool:
    if not text:
        return False
    year = extract_number(text)
    return year is not None and year > 2023


def looks_like_github_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'github.com') and ci_contains(text, 'http')


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
    Restrict evaluator.verify to at most one usage (not used here).
    Favor lenient, fault-tolerant checks and allow partial credit.
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
    paper_info = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperInfo,
        extraction_name="paper_information"
    )

    issue_info = await evaluator.extract(
        prompt=prompt_extract_issue_info(),
        template_class=IssueInfo,
        extraction_name="issue_information"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Google Scholar section
    scholar_node = evaluator.add_sequential(
        id="google_scholar_section",
        desc="Google Scholar search and filtering for LLM Watermarking papers",
        parent=root,
        critical=False
    )

    # [Action Node] scholar.google.com:F3:A6 - Year filtering after 2023
    year_filtering_ok = (
        has_any_ci(answer, ['google scholar', 'scholar']) and
        (has_any_ci(answer, ['2023', '2024', 'after 2023', 'published after', 'since 2023']) or
         looks_like_year_after_2023(paper_info.publication_year))
    )
    evaluator.add_custom_node(
        result=bool(year_filtering_ok),
        id="scholar_action_year_filter",
        desc="[Action Node] scholar.google.com:F3:A6 - Filter papers by publication year (after 2023) using sidebar or search filters",
        parent=scholar_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P1 - Citation count data extraction and judgment
    citation_num = extract_number(paper_info.citation_count_text)
    citation_mentioned = has_any_ci(answer, ['citation', 'cited', 'cites'])
    citation_criteria_ok = citation_num is not None and citation_num >= 50

    evaluator.add_custom_node(
        result=bool(citation_mentioned and citation_criteria_ok),
        id="scholar_perception_citation_count",
        desc="[Perception Node] scholar.google.com:F1:P1 - Extract and verify citation count is at least 50 from search results",
        parent=scholar_node,
        critical=False
    )

    # Additional check: mentions LLM Watermarking topic
    topic_ok = has_any_ci(answer, ['llm watermark', 'watermarking', 'llm', 'language model'])
    evaluator.add_custom_node(
        result=bool(topic_ok),
        id="scholar_mentions_topic",
        desc="Mentions searching for 'LLM Watermarking' or related keywords",
        parent=scholar_node,
        critical=False
    )

    # Additional check: mentions GitHub link requirement
    github_link_search_ok = has_any_ci(answer, ['github', 'code', 'repository', 'runnable'])
    evaluator.add_custom_node(
        result=bool(github_link_search_ok),
        id="scholar_mentions_github_requirement",
        desc="Mentions looking for papers with GitHub links or runnable code",
        parent=scholar_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository navigation and issue list review",
        parent=root,
        critical=False
    )

    # Check if GitHub URL was provided
    github_url_ok = looks_like_github_url(paper_info.github_url)
    evaluator.add_custom_node(
        result=bool(github_url_ok),
        id="github_url_provided",
        desc="GitHub repository URL is provided in the answer",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A8 - Sort or filter operations in issue list
    issue_sorting_ok = has_any_ci(answer, ['issue', 'open issue', 'sort', 'filter', 'browse'])
    evaluator.add_custom_node(
        result=bool(issue_sorting_ok),
        id="github_action_issue_sorting",
        desc="[Action Node] github.com:F4:A8 - Navigate and possibly sort/filter the issue list to review open issues",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F4:P2 - Visual/semantic scanning of issue titles, tags, and status
    error_reports_mentioned = (
        issue_info and issue_info.mentions_error_reports and
        has_any_ci(answer, ['installation', 'environment', 'error', 'failed', 'bug'])
    )
    evaluator.add_custom_node(
        result=bool(error_reports_mentioned),
        id="github_perception_error_reports",
        desc="[Perception Node] github.com:F4:P2 - Scan issue list for error reports like 'Installation failed' or 'Environment error'",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A5 - Pagination to browse issues
    pagination_ok = (
        has_any_ci(answer, ['browse', 'review', 'scan', 'check', 'look through']) and
        has_any_ci(answer, ['issue', 'open issue'])
    )
    evaluator.add_custom_node(
        result=bool(pagination_ok),
        id="github_action_pagination",
        desc="[Action Node] github.com:F4:A5 - Browse through issue list (possibly across pages) to review open issues",
        parent=github_node,
        critical=False
    )

    # Additional check: mentions maintenance quality assessment
    maintenance_ok = (
        issue_info and issue_info.mentions_maintenance_quality and
        has_any_ci(answer, ['maintain', 'best maintained', 'fewest', 'quality', 'complaint'])
    )
    evaluator.add_custom_node(
        result=bool(maintenance_ok),
        id="github_mentions_maintenance",
        desc="Mentions assessing repository maintenance quality and selecting the best maintained one",
        parent=github_node,
        critical=False
    )

    # Additional check: provides open issues count
    open_issues_count = extract_number(issue_info.open_issues_count_text)
    issues_count_ok = open_issues_count is not None
    evaluator.add_custom_node(
        result=bool(issues_count_ok),
        id="github_provides_issue_count",
        desc="Provides the current number of open issues",
        parent=github_node,
        critical=False
    )

    # 3.3 Final output completeness
    output_node = evaluator.add_parallel(
        id="final_output_section",
        desc="Final output completeness check",
        parent=root,
        critical=False
    )

    # Paper title provided
    title_ok = bool(paper_info and paper_info.paper_title and paper_info.paper_title.strip())
    evaluator.add_custom_node(
        result=bool(title_ok),
        id="output_paper_title",
        desc="Provides the paper title",
        parent=output_node,
        critical=False
    )

    # GitHub URL provided
    evaluator.add_custom_node(
        result=bool(github_url_ok),
        id="output_github_url",
        desc="Provides the GitHub repository URL",
        parent=output_node,
        critical=False
    )

    # Open issues count provided
    evaluator.add_custom_node(
        result=bool(issues_count_ok),
        id="output_open_issues_count",
        desc="Provides the current number of open issues",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
