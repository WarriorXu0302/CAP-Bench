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
TASK_ID = "task-1ea4cc"
TASK_DESCRIPTION = 'I need to reproduce a **Music Source Separation** algorithm and want to find several reliable open-source implementations.\n\nFirst, search arXiv for papers on Music Source Separation from the **last six months**, and pick **three papers** that appear to be relatively influential. Then, for each of these three papers, search GitHub for the corresponding official repository.\n\nFor each repository found, focus on checking:\n\n1. Whether it is an **Official Implementation**;  \n2. Whether there have been any code updates (**commits**) in the **last two months**;  \n3. Whether it is a complete project or just a skeleton (e.g., check whether the file tree includes actual `src` or training code, not just README files).\n\nIf no official implementation is found for a paper, look for a relevant community reproduction repository instead (prefer ones with higher star counts) and clearly label it as **“Unofficial.”**\n\nFinally, compile a checklist that includes the paper title, the corresponding GitHub repository link, and conclusions for each of the checks above.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Information about a single paper"""
    paper_title: Optional[str] = None
    paper_authors: Optional[str] = None
    paper_date: Optional[str] = None


class RepositoryInfo(BaseModel):
    """Information about a repository linked to a paper"""
    github_url: Optional[str] = None
    is_official: Optional[str] = None
    has_recent_commits: Optional[str] = None
    is_complete_project: Optional[str] = None


class ChecklistEntry(BaseModel):
    """A single checklist entry"""
    paper_title: Optional[str] = None
    github_url: Optional[str] = None
    is_official: Optional[str] = None
    recent_commits: Optional[str] = None
    project_completeness: Optional[str] = None


class ExtractedChecklist(BaseModel):
    """All checklist entries extracted from the answer"""
    entries: Optional[List[ChecklistEntry]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_papers_from_answer() -> str:
    return """
Extract information about the three papers on Music Source Separation from arXiv that the answer mentions.

For each paper, extract:
- paper_title: the title of the paper
- paper_authors: the authors if mentioned
- paper_date: the publication or submission date if mentioned

Return a list of up to 3 papers. If fewer papers are mentioned, return what is available.
"""


def prompt_extract_checklist_from_answer() -> str:
    return """
Extract the final checklist from the answer that shows the papers and their corresponding GitHub repositories.

For each entry in the checklist, extract:
- paper_title: the paper title
- github_url: the GitHub repository URL
- is_official: whether it is marked as official or unofficial (extract the exact label/conclusion)
- recent_commits: the conclusion about whether there have been commits in the last two months
- project_completeness: the conclusion about whether it is a complete project with src/training code

Return all checklist entries found. If the checklist is missing or incomplete, return what is available.
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


def looks_like_github_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'github.com') or text.startswith('http')


def mentions_arxiv_search(answer: str) -> bool:
    return has_any_ci(answer, ['arxiv'])


def mentions_date_filter(answer: str) -> bool:
    keywords = ['last six months', 'six months', '6 months', 'recent', 'last 6 months']
    return has_any_ci(answer, keywords)


def mentions_github_search(answer: str) -> bool:
    return has_any_ci(answer, ['github'])


def mentions_file_structure(answer: str) -> bool:
    keywords = ['file tree', 'file structure', 'src', 'source', 'training code', 'train', 'directory']
    return has_any_ci(answer, keywords)


def mentions_commits(answer: str) -> bool:
    keywords = ['commit', 'commits', 'update', 'updates', 'last two months', 'two months', '2 months']
    return has_any_ci(answer, keywords)


def mentions_official_check(answer: str) -> bool:
    keywords = ['official', 'unofficial']
    return has_any_ci(answer, keywords)


def count_papers_mentioned(entries: Optional[List[ChecklistEntry]]) -> int:
    if not entries:
        return 0
    unique_titles = set()
    for entry in entries:
        if entry.paper_title and entry.paper_title.strip():
            unique_titles.add(entry.paper_title.strip().lower())
    return len(unique_titles)


def count_repos_with_urls(entries: Optional[List[ChecklistEntry]]) -> int:
    if not entries:
        return 0
    count = 0
    for entry in entries:
        if entry.github_url and looks_like_github_url(entry.github_url):
            count += 1
    return count


def has_official_status_info(entries: Optional[List[ChecklistEntry]]) -> bool:
    if not entries:
        return False
    for entry in entries:
        if entry.is_official and entry.is_official.strip():
            return True
    return False


def has_commits_info(entries: Optional[List[ChecklistEntry]]) -> bool:
    if not entries:
        return False
    for entry in entries:
        if entry.recent_commits and entry.recent_commits.strip():
            return True
    return False


def has_completeness_info(entries: Optional[List[ChecklistEntry]]) -> bool:
    if not entries:
        return False
    for entry in entries:
        if entry.project_completeness and entry.project_completeness.strip():
            return True
    return False


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
    checklist_info = await evaluator.extract(
        prompt=prompt_extract_checklist_from_answer(),
        template_class=ExtractedChecklist,
        extraction_name="checklist_entries"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 arXiv search section
    arxiv_section = evaluator.add_sequential(
        id="arxiv_section",
        desc="Search arXiv for Music Source Separation papers from the last six months",
        parent=root,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A1 - Date filtering
    date_filter_ok = mentions_arxiv_search(answer) and mentions_date_filter(answer)
    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="arxiv_date_filter",
        desc="[Action Node] arxiv.org:F1:A1 - Apply date filter to search for papers from the last six months",
        parent=arxiv_section,
        critical=False
    )

    # Check if three papers are mentioned
    papers_count = count_papers_mentioned(checklist_info.entries if checklist_info else None)
    evaluator.add_custom_node(
        result=bool(papers_count >= 3),
        id="arxiv_three_papers",
        desc="Identify and select three papers on Music Source Separation",
        parent=arxiv_section,
        critical=False
    )

    # 3.2 GitHub search section
    github_section = evaluator.add_sequential(
        id="github_section",
        desc="Search GitHub for repositories corresponding to the selected papers",
        parent=root,
        critical=False
    )

    github_search_ok = mentions_github_search(answer)
    evaluator.add_custom_node(
        result=bool(github_search_ok),
        id="github_search_performed",
        desc="Perform GitHub searches for official or community implementations",
        parent=github_section,
        critical=False
    )

    # Check if GitHub URLs are provided for repos
    repos_count = count_repos_with_urls(checklist_info.entries if checklist_info else None)
    evaluator.add_custom_node(
        result=bool(repos_count >= 1),
        id="github_repos_found",
        desc="Locate at least one GitHub repository link for the papers",
        parent=github_section,
        critical=False
    )

    # 3.3 Repository verification section - Official status
    official_check_section = evaluator.add_sequential(
        id="official_check_section",
        desc="Check whether repositories are official implementations",
        parent=root,
        critical=False
    )

    # [Perception Node] github.com:F3:P23 - README check for official status
    official_mention_ok = mentions_official_check(answer)
    evaluator.add_custom_node(
        result=bool(official_mention_ok),
        id="github_readme_official_check",
        desc="[Perception Node] github.com:F3:P23 - Check README for badges or statements indicating official implementation status",
        parent=official_check_section,
        critical=False
    )

    official_info_ok = has_official_status_info(checklist_info.entries if checklist_info else None)
    evaluator.add_custom_node(
        result=bool(official_info_ok),
        id="official_status_in_checklist",
        desc="Include official/unofficial status in the checklist",
        parent=official_check_section,
        critical=False
    )

    # 3.4 Repository verification section - Recent commits
    commits_check_section = evaluator.add_sequential(
        id="commits_check_section",
        desc="Check for code updates (commits) in the last two months",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F10:P6 - Navigate to commits history
    commits_mention_ok = mentions_commits(answer)
    evaluator.add_custom_node(
        result=bool(commits_mention_ok),
        id="github_commits_navigation",
        desc="[Perception Node] github.com:F10:P6 - Navigate to the commits page and analyze commit history",
        parent=commits_check_section,
        critical=False
    )

    commits_info_ok = has_commits_info(checklist_info.entries if checklist_info else None)
    evaluator.add_custom_node(
        result=bool(commits_info_ok),
        id="recent_commits_in_checklist",
        desc="Include recent commit status in the checklist",
        parent=commits_check_section,
        critical=False
    )

    # 3.5 Repository verification section - Project completeness
    completeness_check_section = evaluator.add_sequential(
        id="completeness_check_section",
        desc="Check whether the repository is a complete project with source/training code",
        parent=root,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - File structure analysis
    file_structure_ok = mentions_file_structure(answer)
    evaluator.add_custom_node(
        result=bool(file_structure_ok),
        id="github_file_structure_check",
        desc="[Perception Node] github.com:F3:P12 - Examine file tree to verify presence of src or training code",
        parent=completeness_check_section,
        critical=False
    )

    completeness_info_ok = has_completeness_info(checklist_info.entries if checklist_info else None)
    evaluator.add_custom_node(
        result=bool(completeness_info_ok),
        id="completeness_in_checklist",
        desc="Include project completeness assessment in the checklist",
        parent=completeness_check_section,
        critical=False
    )

    # 3.6 Final checklist compilation
    checklist_section = evaluator.add_sequential(
        id="checklist_section",
        desc="Compile final checklist with paper titles, GitHub links, and verification results",
        parent=root,
        critical=False
    )

    has_checklist = bool(checklist_info and checklist_info.entries and len(checklist_info.entries) > 0)
    evaluator.add_custom_node(
        result=bool(has_checklist),
        id="checklist_provided",
        desc="Provide a structured checklist with paper and repository information",
        parent=checklist_section,
        critical=False
    )

    # Check if checklist includes all required fields
    all_fields_ok = (has_checklist and
                     papers_count >= 1 and
                     repos_count >= 1 and
                     (has_official_status_info(checklist_info.entries if checklist_info else None) or
                      has_commits_info(checklist_info.entries if checklist_info else None) or
                      has_completeness_info(checklist_info.entries if checklist_info else None)))
    evaluator.add_custom_node(
        result=bool(all_fields_ok),
        id="checklist_completeness",
        desc="Checklist includes paper titles, GitHub URLs, and at least one verification dimension",
        parent=checklist_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
