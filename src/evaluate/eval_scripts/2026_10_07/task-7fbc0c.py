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
TASK_ID = "task-7fbc0c"
TASK_DESCRIPTION = 'I’m working on a data visualization project and want to find a solid Python library. Please search Stack Overflow for **“python data visualization”** and check what libraries are recommended in highly upvoted answers. Record the top three answers with the highest vote counts.\n\nThen look up each of those recommended libraries on GitHub and review their **star counts**, **whether they’ve been updated recently**, and **how clear and complete their README is**. Help me identify which one is the most active.\n\nFinally, search Bilibili for tutorials on that library. Prioritize results by **view count (highest to lowest)**, and find several in-depth tutorials with **high views** and **duration longer than 15 minutes**. Check audience feedback in the comments. If stable sorting by view count is not available, select from the top relevant results and clearly state your selection criteria.\n\nAt the end, tell me whether this library is worth learning and how good the tutorial quality is.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StackOverflowAnswers(BaseModel):
    """Top three answers extracted from Stack Overflow search"""
    answer_1_votes: Optional[str] = None
    answer_1_libraries: Optional[List[str]] = None
    answer_2_votes: Optional[str] = None
    answer_2_libraries: Optional[List[str]] = None
    answer_3_votes: Optional[str] = None
    answer_3_libraries: Optional[List[str]] = None


class GitHubLibraryInfo(BaseModel):
    """GitHub information for recommended libraries"""
    libraries: Optional[List[Dict[str, Any]]] = None
    most_active_library: Optional[str] = None


class BilibiliTutorials(BaseModel):
    """Bilibili tutorial information"""
    tutorials: Optional[List[Dict[str, Any]]] = None
    sorting_criteria: Optional[str] = None
    comment_feedback: Optional[str] = None


class FinalRecommendation(BaseModel):
    """Final recommendation on whether to learn the library"""
    worth_learning: Optional[str] = None
    tutorial_quality: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stackoverflow_answers() -> str:
    return """
Extract the Stack Overflow search results for "python data visualization" from the answer.

Return:
- answer_1_votes: vote count for the first answer
- answer_1_libraries: list of library names mentioned in the first answer
- answer_2_votes: vote count for the second answer
- answer_2_libraries: list of library names mentioned in the second answer
- answer_3_votes: vote count for the third answer
- answer_3_libraries: list of library names mentioned in the third answer

If any field is missing, set it to null or empty list.
"""


def prompt_extract_github_info() -> str:
    return """
From the answer, extract the GitHub information for the recommended libraries:

- libraries: a list of dictionaries, each containing:
  - name: library name
  - stars: star count
  - recently_updated: whether recently updated (true/false or descriptive text)
  - readme_quality: assessment of README clarity and completeness
- most_active_library: the name of the library identified as most active

If any field is missing, set it to null.
"""


def prompt_extract_bilibili_tutorials() -> str:
    return """
From the answer, extract the Bilibili tutorial information:

- tutorials: a list of dictionaries, each containing:
  - title: tutorial title
  - views: view count
  - duration: video duration
  - comments_summary: summary of audience feedback
- sorting_criteria: explanation of how results were sorted or selected
- comment_feedback: overall assessment of comment feedback quality

If any field is missing, set it to null.
"""


def prompt_extract_final_recommendation() -> str:
    return """
From the answer, extract the final recommendation:

- worth_learning: whether the library is worth learning (yes/no or detailed explanation)
- tutorial_quality: assessment of tutorial quality

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
    matches = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not matches:
        return None
    try:
        return float(matches[0].replace(',', ''))
    except Exception:
        return None


def mentions_stackoverflow(answer: str) -> bool:
    return has_any_ci(answer, ['stack overflow', 'stackoverflow'])


def mentions_sorting_by_votes(answer: str) -> bool:
    return has_any_ci(answer, ['vote', 'upvote', 'highest vote', 'most vote', 'sorted by vote'])


def mentions_github(answer: str) -> bool:
    return has_any_ci(answer, ['github'])


def mentions_stars(answer: str) -> bool:
    return has_any_ci(answer, ['star', 'starred'])


def mentions_recent_updates(answer: str) -> bool:
    return has_any_ci(answer, ['recent', 'update', 'commit', 'active', 'maintained'])


def mentions_readme(answer: str) -> bool:
    return has_any_ci(answer, ['readme', 'documentation', 'docs'])


def mentions_bilibili(answer: str) -> bool:
    return has_any_ci(answer, ['bilibili', 'b站'])


def mentions_view_count(answer: str) -> bool:
    return has_any_ci(answer, ['view', '播放', '播放量'])


def mentions_duration(answer: str) -> bool:
    return has_any_ci(answer, ['duration', 'minute', '分钟', 'long', '15 minute', '15分钟'])


def mentions_comments(answer: str) -> bool:
    return has_any_ci(answer, ['comment', 'feedback', '评论', '反馈'])


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
    stackoverflow_info = await evaluator.extract(
        prompt=prompt_extract_stackoverflow_answers(),
        template_class=StackOverflowAnswers,
        extraction_name="stackoverflow_answers"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_info(),
        template_class=GitHubLibraryInfo,
        extraction_name="github_library_info"
    )

    bilibili_info = await evaluator.extract(
        prompt=prompt_extract_bilibili_tutorials(),
        template_class=BilibiliTutorials,
        extraction_name="bilibili_tutorials"
    )

    final_rec = await evaluator.extract(
        prompt=prompt_extract_final_recommendation(),
        template_class=FinalRecommendation,
        extraction_name="final_recommendation"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Stack Overflow section
    stackoverflow_node = evaluator.add_sequential(
        id="stackoverflow_section",
        desc="Stack Overflow search and analysis for python data visualization",
        parent=root,
        critical=False
    )

    # [Action Node] stackoverflow.com:F2:A2 - Sort by votes
    stackoverflow_sorted_ok = mentions_stackoverflow(answer) and mentions_sorting_by_votes(answer)
    evaluator.add_custom_node(
        result=bool(stackoverflow_sorted_ok),
        id="stackoverflow_action_sort_by_votes",
        desc="[Action Node] stackoverflow.com:F2:A2 - Search Stack Overflow and sort results by vote count",
        parent=stackoverflow_node,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F2:P9 - Understand answer content and extract library names
    has_libraries = False
    if stackoverflow_info:
        libs = []
        if stackoverflow_info.answer_1_libraries:
            libs.extend(stackoverflow_info.answer_1_libraries)
        if stackoverflow_info.answer_2_libraries:
            libs.extend(stackoverflow_info.answer_2_libraries)
        if stackoverflow_info.answer_3_libraries:
            libs.extend(stackoverflow_info.answer_3_libraries)
        has_libraries = len(libs) > 0

    evaluator.add_custom_node(
        result=bool(has_libraries),
        id="stackoverflow_perception_extract_libraries",
        desc="[Perception Node] stackoverflow.com:F2:P9 - Understand answer content and extract recommended library names",
        parent=stackoverflow_node,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F4:P14 - Recognize code examples and library names
    has_code_understanding = has_libraries or has_any_ci(answer, ['import', 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'altair'])
    evaluator.add_custom_node(
        result=bool(has_code_understanding),
        id="stackoverflow_perception_code_understanding",
        desc="[Perception Node] stackoverflow.com:F4:P14 - Recognize code examples and library names in answers",
        parent=stackoverflow_node,
        critical=False
    )

    # Check for top three answers
    has_top_three = False
    if stackoverflow_info:
        has_answer_1 = bool(stackoverflow_info.answer_1_votes and stackoverflow_info.answer_1_libraries)
        has_answer_2 = bool(stackoverflow_info.answer_2_votes and stackoverflow_info.answer_2_libraries)
        has_answer_3 = bool(stackoverflow_info.answer_3_votes and stackoverflow_info.answer_3_libraries)
        has_top_three = has_answer_1 and has_answer_2 and has_answer_3

    evaluator.add_custom_node(
        result=bool(has_top_three),
        id="stackoverflow_has_top_three_answers",
        desc="Records top three answers with vote counts",
        parent=stackoverflow_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub analysis of recommended libraries",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by star count
    github_stars_mentioned = mentions_github(answer) and mentions_stars(answer)
    evaluator.add_custom_node(
        result=bool(github_stars_mentioned),
        id="github_action_check_stars",
        desc="[Action Node] github.com:F1:A7 - Check repositories and consider star counts",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Extract star count and update status
    has_star_info = False
    has_update_info = False
    if github_info and github_info.libraries:
        for lib in github_info.libraries:
            if isinstance(lib, dict):
                if lib.get('stars'):
                    has_star_info = True
                if lib.get('recently_updated') is not None:
                    has_update_info = True

    evaluator.add_custom_node(
        result=bool(has_star_info and has_update_info),
        id="github_perception_extract_metrics",
        desc="[Perception Node] github.com:F1:P1 - Extract star counts and recent update status for libraries",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P10 - Identify project activity status
    has_activity_analysis = mentions_recent_updates(answer) or has_update_info
    evaluator.add_custom_node(
        result=bool(has_activity_analysis),
        id="github_perception_activity_status",
        desc="[Perception Node] github.com:F1:P10 - Identify whether projects are archived and update frequency",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P12 - Understand README structure and completeness
    has_readme_analysis = False
    if github_info and github_info.libraries:
        for lib in github_info.libraries:
            if isinstance(lib, dict) and lib.get('readme_quality'):
                has_readme_analysis = True
                break

    readme_mentioned = mentions_readme(answer)
    evaluator.add_custom_node(
        result=bool(has_readme_analysis or readme_mentioned),
        id="github_perception_readme_quality",
        desc="[Perception Node] github.com:F3:P12 - Understand and assess README clarity and completeness",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F10:P6 - Check commit history for recent updates
    has_commit_check = mentions_recent_updates(answer) and has_any_ci(answer, ['commit', 'update', 'recent'])
    evaluator.add_custom_node(
        result=bool(has_commit_check),
        id="github_perception_commit_history",
        desc="[Perception Node] github.com:F10:P6 - Review commit history to verify recent updates",
        parent=github_node,
        critical=False
    )

    # Identify most active library
    has_most_active = bool(github_info and github_info.most_active_library)
    evaluator.add_custom_node(
        result=bool(has_most_active),
        id="github_identifies_most_active",
        desc="Identifies which library is the most active",
        parent=github_node,
        critical=False
    )

    # 3.3 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili tutorial search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sort by view count
    bilibili_sorted = mentions_bilibili(answer) and mentions_view_count(answer)
    evaluator.add_custom_node(
        result=bool(bilibili_sorted),
        id="bilibili_action_sort_by_views",
        desc="[Action Node] bilibili.com:F1:A5 - Search Bilibili and sort/prioritize by view count",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Recognize thumbnails to judge content relevance
    has_thumbnail_understanding = mentions_bilibili(answer) and has_any_ci(answer, ['tutorial', '教程', 'video', '视频'])
    evaluator.add_custom_node(
        result=bool(has_thumbnail_understanding),
        id="bilibili_perception_thumbnail_relevance",
        desc="[Perception Node] bilibili.com:F1:P30 - Recognize search result thumbnails to judge content relevance",
        parent=bilibili_node,
        critical=False
    )

    # Check for tutorials with high views and long duration
    has_qualifying_tutorials = False
    if bilibili_info and bilibili_info.tutorials:
        for tutorial in bilibili_info.tutorials:
            if isinstance(tutorial, dict):
                has_views = tutorial.get('views') is not None
                has_duration = tutorial.get('duration') is not None
                if has_views and has_duration:
                    has_qualifying_tutorials = True
                    break

    duration_mentioned = mentions_duration(answer)
    evaluator.add_custom_node(
        result=bool(has_qualifying_tutorials or (duration_mentioned and mentions_view_count(answer))),
        id="bilibili_has_qualifying_tutorials",
        desc="Finds tutorials with high views and duration longer than 15 minutes",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F5:A44 - Scroll to load more comments
    comments_checked = mentions_comments(answer)
    evaluator.add_custom_node(
        result=bool(comments_checked),
        id="bilibili_action_load_comments",
        desc="[Action Node] bilibili.com:F5:A44 - Scroll to load and view comments",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F5:P19 - Understand comment content and user feedback
    has_comment_analysis = False
    if bilibili_info:
        has_comment_analysis = bool(bilibili_info.comment_feedback or
                                   (bilibili_info.tutorials and
                                    any(isinstance(t, dict) and t.get('comments_summary')
                                        for t in bilibili_info.tutorials or [])))

    evaluator.add_custom_node(
        result=bool(has_comment_analysis),
        id="bilibili_perception_comment_understanding",
        desc="[Perception Node] bilibili.com:F5:P19 - Understand comment content and user feedback quality",
        parent=bilibili_node,
        critical=False
    )

    # Check for sorting criteria explanation
    has_sorting_explanation = bool(bilibili_info and bilibili_info.sorting_criteria)
    evaluator.add_custom_node(
        result=bool(has_sorting_explanation),
        id="bilibili_explains_sorting_criteria",
        desc="Clearly states the selection or sorting criteria used",
        parent=bilibili_node,
        critical=False
    )

    # 3.4 Final recommendation section
    final_node = evaluator.add_sequential(
        id="final_recommendation_section",
        desc="Final recommendation on library and tutorial quality",
        parent=root,
        critical=False
    )

    has_worth_learning = bool(final_rec and final_rec.worth_learning)
    evaluator.add_custom_node(
        result=bool(has_worth_learning),
        id="final_worth_learning_assessment",
        desc="Provides assessment on whether the library is worth learning",
        parent=final_node,
        critical=False
    )

    has_tutorial_quality = bool(final_rec and final_rec.tutorial_quality)
    evaluator.add_custom_node(
        result=bool(has_tutorial_quality),
        id="final_tutorial_quality_assessment",
        desc="Provides assessment on tutorial quality",
        parent=final_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
