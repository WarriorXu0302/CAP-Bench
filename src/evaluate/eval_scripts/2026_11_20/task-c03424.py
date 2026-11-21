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
TASK_ID = "task-c03424"
TASK_DESCRIPTION = "I want to reproduce a high-quality paper on Image Inpainting.\n\nFirst, please search Semantic Scholar for me. Find conference papers on 'Image Inpainting' published between 2020 and 2023. Sort them by citation count (descending) and identify the most cited paper. Tell me its title and the number of citations.\n\nNext, find the corresponding GitHub repository for this paper. I'd like to know its approximate star count and if it's a popular project with thousands of stars.\n\nFinally, to build a foundational understanding, please find the highest-rated course related to GANs (Generative Adversarial Networks) on Coursera. Provide me with the course name and its rating. Ideally, I'd prefer an 'Advanced' level course; if no 'Advanced' options are available, then an 'Intermediate' one will suffice."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SemanticScholarPaperInfo(BaseModel):
    """Most cited paper information from Semantic Scholar"""
    paper_title: Optional[str] = None
    citation_count: Optional[str] = None


class GitHubRepoInfo(BaseModel):
    """GitHub repository information for the paper"""
    repo_name: Optional[str] = None
    star_count_text: Optional[str] = None


class CourseraGANCourseInfo(BaseModel):
    """Coursera GAN course information"""
    course_name: Optional[str] = None
    course_rating: Optional[str] = None
    course_level: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_semantic_scholar_paper() -> str:
    return """
Extract the most cited conference paper on 'Image Inpainting' (2020-2023) that the user found on Semantic Scholar.

Return:
- paper_title: the exact title of the paper as stated in the answer.
- citation_count: the number of citations exactly as written (include commas or any formatting if present).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_github_repo() -> str:
    return """
From the answer, extract the GitHub repository information for the paper:

- repo_name: the repository name or URL if mentioned.
- star_count_text: the approximate star count exactly as stated (include "k" notation or any formatting if present).

If any field is missing, set it to null.
"""


def prompt_extract_coursera_course() -> str:
    return """
From the answer, extract the highest-rated Coursera course on GANs:

- course_name: the exact course name as stated.
- course_rating: the rating exactly as written (include "/5" or any formatting if present).
- course_level: the difficulty level (Advanced, Intermediate, Beginner, etc.) if mentioned.

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


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Handle formats like "1,234", "1.5k", "1234", etc.
    cleaned = re.sub(r'[,\s]', '', text.lower())
    m = re.search(r'(\d+(?:\.\d+)?)\s*k', cleaned)
    if m:
        try:
            return float(m.group(1)) * 1000
        except Exception:
            pass
    m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def looks_like_citation_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


def looks_like_star_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_number(text)
    if num is None:
        return False
    # Typical rating range is 0-5
    return 0 <= num <= 5


def mentions_thousands_of_stars(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['thousand', 'thousands', 'k stars', 'k star', 'popular'])


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
        prompt=prompt_extract_semantic_scholar_paper(),
        template_class=SemanticScholarPaperInfo,
        extraction_name="semantic_scholar_paper"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_repo(),
        template_class=GitHubRepoInfo,
        extraction_name="github_repository"
    )

    coursera_info = await evaluator.extract(
        prompt=prompt_extract_coursera_course(),
        template_class=CourseraGANCourseInfo,
        extraction_name="coursera_gan_course"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Semantic Scholar part
    semantic_scholar_node = evaluator.add_sequential(
        id="semantic_scholar_section",
        desc="Semantic Scholar search for most cited Image Inpainting conference paper (2020-2023)",
        parent=root,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A2 - Publication Type filter (conference papers)
    ss_action_publication_type = (has_any_ci(answer, ['semantic scholar']) and
                                   has_any_ci(answer, ['conference']))
    evaluator.add_custom_node(
        result=bool(ss_action_publication_type),
        id="ss_action_publication_type_filter",
        desc="[Action Node] semanticscholar.org:F1:A2 - Filter by Publication Type (conference papers)",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A3 - Date Range filter (2020-2023)
    ss_action_date_range = (has_any_ci(answer, ['2020', '2023']) or
                            (has_any_ci(answer, ['2020']) and has_any_ci(answer, ['2023'])))
    evaluator.add_custom_node(
        result=bool(ss_action_date_range),
        id="ss_action_date_range_filter",
        desc="[Action Node] semanticscholar.org:F1:A3 - Filter by Date Range (2020-2023)",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Sort by citation count
    ss_action_sort = (has_any_ci(answer, ['citation', 'citations', 'cited']) and
                      has_any_ci(answer, ['sort', 'sorted', 'descending', 'most']))
    evaluator.add_custom_node(
        result=bool(ss_action_sort),
        id="ss_action_sort_by_citations",
        desc="[Action Node] semanticscholar.org:F1:A4 - Sort results by citation count (descending)",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F1:P3 - Extract paper title and citation count
    paper_title_ok = bool(paper_info and paper_info.paper_title and paper_info.paper_title.strip())
    citation_count_ok = looks_like_citation_count(paper_info.citation_count)

    evaluator.add_custom_node(
        result=bool(paper_title_ok and citation_count_ok),
        id="ss_perception_paper_info",
        desc="[Perception Node] semanticscholar.org:F1:P3 - Extract the most cited paper's title and citation count",
        parent=semantic_scholar_node,
        critical=False
    )

    # Mentions Image Inpainting context
    mentions_inpainting = has_any_ci(answer, ['image inpainting', 'inpainting'])
    evaluator.add_custom_node(
        result=bool(mentions_inpainting),
        id="ss_mentions_image_inpainting",
        desc="Mentions 'Image Inpainting' as the search topic",
        parent=semantic_scholar_node,
        critical=False
    )

    # 3.2 GitHub part
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository lookup for the paper",
        parent=root,
        critical=False
    )

    # Action: finding the GitHub repo
    github_action_ok = (has_any_ci(answer, ['github']) and
                        (has_any_ci(answer, ['repository', 'repo']) or has_any_ci(answer, ['star'])))
    evaluator.add_custom_node(
        result=bool(github_action_ok),
        id="github_action_find_repo",
        desc="Navigate to GitHub and locate the repository for the paper",
        parent=github_node,
        critical=False
    )

    # Perception: extract star count
    repo_name_ok = bool(github_info and github_info.repo_name and github_info.repo_name.strip())
    star_count_ok = looks_like_star_count(github_info.star_count_text)

    evaluator.add_custom_node(
        result=bool(star_count_ok),
        id="github_perception_star_count",
        desc="Extract the approximate star count from the GitHub repository",
        parent=github_node,
        critical=False
    )

    # Check if mentions popularity/thousands of stars
    popularity_mentioned = mentions_thousands_of_stars(answer)
    evaluator.add_custom_node(
        result=bool(popularity_mentioned),
        id="github_mentions_popularity",
        desc="Mentions if the project is popular with thousands of stars",
        parent=github_node,
        critical=False
    )

    # 3.3 Coursera part
    coursera_node = evaluator.add_sequential(
        id="coursera_section",
        desc="Coursera search for highest-rated GAN course (preferably Advanced or Intermediate)",
        parent=root,
        critical=False
    )

    # Action: search Coursera for GAN courses
    coursera_action_ok = (has_any_ci(answer, ['coursera']) and
                          has_any_ci(answer, ['gan', 'generative adversarial network']))
    evaluator.add_custom_node(
        result=bool(coursera_action_ok),
        id="coursera_action_search_gan",
        desc="Search Coursera for GAN-related courses",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F1:A12 - Sort by rating
    coursera_action_sort = (has_any_ci(answer, ['highest-rated', 'highest rated', 'rating', 'rated']) and
                            has_any_ci(answer, ['sort', 'sorted']))
    evaluator.add_custom_node(
        result=bool(coursera_action_sort),
        id="coursera_action_sort_by_rating",
        desc="[Action Node] coursera.org:F1:A12 - Sort courses by rating to find the highest-rated one",
        parent=coursera_node,
        critical=False
    )

    # Perception: extract course name and rating
    course_name_ok = bool(coursera_info and coursera_info.course_name and coursera_info.course_name.strip())
    rating_ok = looks_like_rating(coursera_info.course_rating)

    evaluator.add_custom_node(
        result=bool(course_name_ok and rating_ok),
        id="coursera_perception_course_info",
        desc="Extract the course name and rating",
        parent=coursera_node,
        critical=False
    )

    # Check course level preference (Advanced > Intermediate)
    level_mentioned = bool(coursera_info and coursera_info.course_level and coursera_info.course_level.strip())
    level_appropriate = (level_mentioned and
                         has_any_ci(coursera_info.course_level, ['advanced', 'intermediate']))
    evaluator.add_custom_node(
        result=bool(level_appropriate),
        id="coursera_level_preference",
        desc="Course level is Advanced or Intermediate as preferred",
        parent=coursera_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
