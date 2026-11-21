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
TASK_ID = "task-d2bd5f"
TASK_DESCRIPTION = 'I’m taking the SQL course on Khan Academy.  \nFirst, check the course page for any recommended learning resources and tools, and note down any **recommended books** (if there are no explicit book recommendations on the course page, clearly record **“No explicit book recommendations found”** and continue with the next steps).\n\nThen, go to Amazon and look up those books, checking their **prices** and **ratings**. Record the book(s) with the **best reviews** and priced **at or below $50** (if none meet both criteria, record the closest candidates and specify which criteria are not met).\n\nIn addition, I want some companion SQL hands-on projects for practice. Go to GitHub and search for **“SQL practice projects”**, sort by **stars**, and review the top results across the first few pages to find beginner-friendly projects. Open each project and check the README to confirm it includes a **complete tutorial** and a **practice dataset**.\n\nFinally, compile a checklist for me, including the **books worth buying** and **recommended practice project links**.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class KhanAcademyBooks(BaseModel):
    """Book recommendations extracted from Khan Academy course page"""
    books_found: Optional[bool] = None
    book_titles: Optional[List[str]] = Field(default_factory=list)
    explicit_no_books_statement: Optional[bool] = None


class AmazonBookInfo(BaseModel):
    """Amazon book search results with prices and ratings"""
    books_checked: Optional[List[str]] = Field(default_factory=list)
    books_with_price_and_rating: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    books_meeting_criteria: Optional[List[str]] = Field(default_factory=list)
    books_not_meeting_criteria_explanation: Optional[str] = None


class GitHubProjects(BaseModel):
    """GitHub SQL practice projects information"""
    search_performed: Optional[bool] = None
    sorted_by_stars: Optional[bool] = None
    projects_reviewed: Optional[List[str]] = Field(default_factory=list)
    projects_with_tutorial_and_dataset: Optional[List[str]] = Field(default_factory=list)


class FinalChecklist(BaseModel):
    """Final compiled checklist"""
    books_worth_buying: Optional[List[str]] = Field(default_factory=list)
    recommended_project_links: Optional[List[str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_khanacademy_books() -> str:
    return """
Extract information about book recommendations from the Khan Academy SQL course page mentioned in the answer.

Return:
- books_found: true if any books were found/mentioned, false otherwise
- book_titles: list of book titles mentioned (empty list if none)
- explicit_no_books_statement: true if the answer explicitly states "No explicit book recommendations found" or similar

If information is missing, set fields to null or empty lists as appropriate.
"""


def prompt_extract_amazon_books() -> str:
    return """
Extract information about Amazon book searches from the answer.

Return:
- books_checked: list of book titles that were looked up on Amazon
- books_with_price_and_rating: list of dicts with book info including price and rating
- books_meeting_criteria: list of books that meet the criteria (best reviews AND <= $50)
- books_not_meeting_criteria_explanation: explanation if no books meet both criteria

If information is missing, set fields to null or empty lists.
"""


def prompt_extract_github_projects() -> str:
    return """
Extract information about GitHub SQL practice project searches from the answer.

Return:
- search_performed: true if GitHub search was performed
- sorted_by_stars: true if results were sorted by stars
- projects_reviewed: list of project names/repos reviewed
- projects_with_tutorial_and_dataset: list of projects confirmed to have complete tutorial AND practice dataset

If information is missing, set to null or empty lists.
"""


def prompt_extract_final_checklist() -> str:
    return """
Extract the final compiled checklist from the answer.

Return:
- books_worth_buying: list of book titles recommended for purchase
- recommended_project_links: list of GitHub project links/names recommended

If information is missing, set to empty lists.
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


def contains_url_or_link(text: Optional[str], domain: str = "") -> bool:
    if not text:
        return False
    if domain:
        return ci_contains(text, domain) and (ci_contains(text, 'http') or ci_contains(text, 'www') or ci_contains(text, '.com'))
    return ci_contains(text, 'http') or ci_contains(text, 'www')


def has_price_info(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'\$\d+|\d+\.\d{2}|price', text, re.IGNORECASE))


def has_rating_info(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'rating|stars?|\d+\.\d+\s*(out of|\/)\s*5|review', text, re.IGNORECASE))


def mentions_price_limit(text: Optional[str], limit: int = 50) -> bool:
    if not text:
        return False
    return bool(re.search(rf'\${limit}|{limit}\s*dollar|at or below|under \${limit}|<\s*\${limit}', text, re.IGNORECASE))


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
    khanacademy_info = await evaluator.extract(
        prompt=prompt_extract_khanacademy_books(),
        template_class=KhanAcademyBooks,
        extraction_name="khanacademy_books"
    )

    amazon_info = await evaluator.extract(
        prompt=prompt_extract_amazon_books(),
        template_class=AmazonBookInfo,
        extraction_name="amazon_books"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_projects(),
        template_class=GitHubProjects,
        extraction_name="github_projects"
    )

    checklist_info = await evaluator.extract(
        prompt=prompt_extract_final_checklist(),
        template_class=FinalChecklist,
        extraction_name="final_checklist"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Khan Academy section
    khanacademy_node = evaluator.add_sequential(
        id="khanacademy_section",
        desc="Khan Academy SQL course - Check for recommended books",
        parent=root,
        critical=False
    )

    # [Action Node] khanacademy.org:F2:A19 - Expand lesson panel to view resources
    expand_action_ok = (
        ci_contains(answer, 'khan academy') and
        has_any_ci(answer, ['lesson', 'resource', 'expand', 'panel', 'course page', 'recommend'])
    )
    evaluator.add_custom_node(
        result=bool(expand_action_ok),
        id="khanacademy_expand_panel",
        desc="[Action Node] khanacademy.org:F2:A19 - Expand lesson/course panel to view recommended resources",
        parent=khanacademy_node,
        critical=False
    )

    # [Perception Node] khanacademy.org:F2:P7 - Understand and identify book recommendations
    books_identified = (
        khanacademy_info and
        (
            (khanacademy_info.books_found and len(khanacademy_info.book_titles) > 0) or
            khanacademy_info.explicit_no_books_statement
        )
    )
    evaluator.add_custom_node(
        result=bool(books_identified),
        id="khanacademy_identify_books",
        desc='[Perception Node] khanacademy.org:F2:P7 - Identify book recommendations or explicitly note "No explicit book recommendations found"',
        parent=khanacademy_node,
        critical=False
    )

    # 3.2 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon - Search books, check prices and ratings",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A5 - Select Books category from dropdown
    amazon_category_ok = (
        ci_contains(answer, 'amazon') and
        has_any_ci(answer, ['book', 'search'])
    )
    evaluator.add_custom_node(
        result=bool(amazon_category_ok),
        id="amazon_select_books_category",
        desc="[Action Node] Amazon:F1:A5 - Search for books on Amazon (may involve category selection)",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F1:A6 - Navigate through search result pages
    amazon_pagination_ok = (
        amazon_info and
        amazon_info.books_checked and
        len(amazon_info.books_checked) > 0
    )
    evaluator.add_custom_node(
        result=bool(amazon_pagination_ok),
        id="amazon_pagination",
        desc="[Action Node] Amazon:F1:A6 - Navigate through search results to check multiple books",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P3 - Identify book covers/products
    amazon_image_ok = (
        ci_contains(answer, 'amazon') and
        has_any_ci(answer, ['book', 'cover', 'product', 'result'])
    )
    evaluator.add_custom_node(
        result=bool(amazon_image_ok),
        id="amazon_identify_books",
        desc="[Perception Node] Amazon:F5:P3 - Identify correct books from search results (image understanding)",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F10:P15 - Understand reviews to determine best reviewed books
    amazon_reviews_ok = (
        amazon_info and
        (
            (amazon_info.books_with_price_and_rating and len(amazon_info.books_with_price_and_rating) > 0) or
            has_rating_info(answer)
        ) and
        has_any_ci(answer, ['review', 'rating', 'best'])
    )
    evaluator.add_custom_node(
        result=bool(amazon_reviews_ok),
        id="amazon_understand_reviews",
        desc="[Perception Node] Amazon:F10:P15 - Understand and analyze reviews to identify best reviewed books",
        parent=amazon_node,
        critical=False
    )

    # Check price and rating extraction
    price_rating_extracted = (
        amazon_info and
        has_price_info(answer) and
        has_rating_info(answer)
    )
    evaluator.add_custom_node(
        result=bool(price_rating_extracted),
        id="amazon_extract_price_rating",
        desc="Extract price and rating information for books",
        parent=amazon_node,
        critical=False
    )

    # Check $50 criteria application
    criteria_applied = (
        amazon_info and
        (
            (amazon_info.books_meeting_criteria and len(amazon_info.books_meeting_criteria) > 0) or
            amazon_info.books_not_meeting_criteria_explanation
        ) and
        mentions_price_limit(answer, 50)
    )
    evaluator.add_custom_node(
        result=bool(criteria_applied),
        id="amazon_apply_criteria",
        desc="Apply criteria: best reviews AND price at or below $50",
        parent=amazon_node,
        critical=False
    )

    # 3.3 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub - Search SQL practice projects, sort by stars, review READMEs",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by stars using dropdown
    github_sort_ok = (
        github_info and
        github_info.sorted_by_stars and
        has_any_ci(answer, ['github', 'stars', 'sort'])
    )
    evaluator.add_custom_node(
        result=bool(github_sort_ok),
        id="github_sort_by_stars",
        desc="[Action Node] github.com:F1:A7 - Sort search results by stars (Most stars)",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A4 - Navigate through multiple pages
    github_pagination_ok = (
        github_info and
        github_info.projects_reviewed and
        len(github_info.projects_reviewed) > 1 and
        has_any_ci(answer, ['page', 'first few pages', 'multiple'])
    )
    evaluator.add_custom_node(
        result=bool(github_pagination_ok),
        id="github_pagination",
        desc="[Action Node] github.com:F1:A4 - Navigate through first few pages of search results",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Extract project information from search results
    github_extract_ok = (
        github_info and
        github_info.projects_reviewed and
        len(github_info.projects_reviewed) > 0
    )
    evaluator.add_custom_node(
        result=bool(github_extract_ok),
        id="github_extract_project_info",
        desc="[Perception Node] github.com:F1:P1 - Extract project names, stars, descriptions from search results",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F3:A1 - Navigate to project README (Code tab)
    github_readme_navigation_ok = (
        has_any_ci(answer, ['readme', 'open', 'click', 'check', 'project']) and
        github_info and
        github_info.projects_reviewed and
        len(github_info.projects_reviewed) > 0
    )
    evaluator.add_custom_node(
        result=bool(github_readme_navigation_ok),
        id="github_navigate_to_readme",
        desc="[Action Node] github.com:F3:A1 - Navigate to project pages to view README",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P22 - Read and understand README text
    github_readme_text_ok = (
        github_info and
        github_info.projects_with_tutorial_and_dataset and
        len(github_info.projects_with_tutorial_and_dataset) > 0 and
        has_any_ci(answer, ['readme', 'tutorial', 'dataset'])
    )
    evaluator.add_custom_node(
        result=bool(github_readme_text_ok),
        id="github_understand_readme_text",
        desc="[Perception Node] github.com:F3:P22 - Read and understand README to confirm tutorial and dataset presence",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P23 - Understand diagrams/screenshots if present
    github_scene_understanding_ok = (
        has_any_ci(answer, ['readme', 'diagram', 'screenshot', 'architecture', 'demo']) or
        (github_info and github_info.projects_with_tutorial_and_dataset and len(github_info.projects_with_tutorial_and_dataset) > 0)
    )
    evaluator.add_custom_node(
        result=bool(github_scene_understanding_ok),
        id="github_scene_understanding",
        desc="[Perception Node] github.com:F3:P23 - Understand project architecture diagrams or demo screenshots in README if present",
        parent=github_node,
        critical=False
    )

    # Check for beginner-friendly filter
    beginner_friendly_ok = has_any_ci(answer, ['beginner', 'beginner-friendly', 'easy', 'starter', 'introductory'])
    evaluator.add_custom_node(
        result=bool(beginner_friendly_ok),
        id="github_beginner_friendly",
        desc="Review projects for beginner-friendly characteristics",
        parent=github_node,
        critical=False
    )

    # 3.4 Final checklist compilation
    checklist_node = evaluator.add_parallel(
        id="final_checklist_section",
        desc="Compile final checklist with books and projects",
        parent=root,
        critical=False
    )

    # Books worth buying in checklist
    books_in_checklist = (
        checklist_info and
        checklist_info.books_worth_buying and
        len(checklist_info.books_worth_buying) > 0
    ) or has_any_ci(answer, ['books worth buying', 'recommend', 'purchase'])
    evaluator.add_custom_node(
        result=bool(books_in_checklist),
        id="checklist_books",
        desc="Final checklist includes books worth buying",
        parent=checklist_node,
        critical=False
    )

    # Project links in checklist
    projects_in_checklist = (
        checklist_info and
        checklist_info.recommended_project_links and
        len(checklist_info.recommended_project_links) > 0
    ) or (contains_url_or_link(answer, 'github') and has_any_ci(answer, ['project', 'link', 'recommend']))
    evaluator.add_custom_node(
        result=bool(projects_in_checklist),
        id="checklist_projects",
        desc="Final checklist includes recommended practice project links",
        parent=checklist_node,
        critical=False
    )

    # Checklist format and organization
    checklist_formatted = (
        has_any_ci(answer, ['checklist', 'list', 'summary', 'recommendation']) and
        (books_in_checklist or projects_in_checklist)
    )
    evaluator.add_custom_node(
        result=bool(checklist_formatted),
        id="checklist_format",
        desc="Final output is organized as a clear checklist",
        parent=checklist_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
