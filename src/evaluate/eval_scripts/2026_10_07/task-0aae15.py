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
TASK_ID = "task-0aae15"
TASK_DESCRIPTION = "I want to contribute code to the pandas Python library but don't know where to begin.\n\nFirst, help me find open issues in the pandas project on GitHub that are labeled 'good first issue'. From those, identify the top 5 most active issues (those with recent updates and many comments), and for each, record its title, a brief summary of the topic, and the number of discussions.\n\nNext, search for pandas-related questions on StackOverflow, sorted by votes, to find the most frequently asked and highly upvoted ones. List the titles and vote counts for the top 10.\n\nFinally, compare these high-frequency StackOverflow questions with the themes of the GitHub issues. Identify issues that are suitable for beginners and simultaneously address significant user pain points. Recommend 2-3 of the most worthwhile issues for me to contribute to."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GitHubIssue(BaseModel):
    """A single GitHub issue extracted from the answer"""
    title: Optional[str] = None
    summary: Optional[str] = None
    discussion_count: Optional[int] = None


class GitHubIssuesData(BaseModel):
    """GitHub issues data extracted from the answer"""
    issues: List[GitHubIssue] = Field(default_factory=list)


class StackOverflowQuestion(BaseModel):
    """A single StackOverflow question extracted from the answer"""
    title: Optional[str] = None
    vote_count: Optional[int] = None


class StackOverflowQuestionsData(BaseModel):
    """StackOverflow questions data extracted from the answer"""
    questions: List[StackOverflowQuestion] = Field(default_factory=list)


class RecommendationsData(BaseModel):
    """Recommendations extracted from the answer"""
    recommended_issues: List[str] = Field(default_factory=list)
    comparison_performed: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_github_issues() -> str:
    return """
Extract the GitHub issues information from the answer. The user was asked to find the top 5 most active 'good first issue' labeled issues in the pandas project.

For each issue mentioned, extract:
- title: the issue title as stated
- summary: a brief summary of what the issue is about
- discussion_count: the number of discussions/comments mentioned

Return a list of issues. If fewer than 5 issues are mentioned, return what is available. If no issues are found, return an empty list.
"""


def prompt_extract_stackoverflow_questions() -> str:
    return """
Extract the StackOverflow questions information from the answer. The user was asked to find the top 10 pandas-related questions sorted by votes.

For each question mentioned, extract:
- title: the question title as stated
- vote_count: the vote count mentioned

Return a list of questions. If fewer than 10 questions are mentioned, return what is available. If no questions are found, return an empty list.
"""


def prompt_extract_recommendations() -> str:
    return """
Extract the final recommendations from the answer. The user was asked to compare StackOverflow questions with GitHub issues and recommend 2-3 worthwhile issues.

Extract:
- recommended_issues: a list of issue titles or descriptions that were recommended
- comparison_performed: true if the answer shows evidence of comparing StackOverflow questions with GitHub issues, false otherwise

If no recommendations are found, return empty list for recommended_issues.
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


def count_non_empty(items: List[Any]) -> int:
    return sum(1 for item in items if item)


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
    github_data = await evaluator.extract(
        prompt=prompt_extract_github_issues(),
        template_class=GitHubIssuesData,
        extraction_name="github_issues"
    )

    stackoverflow_data = await evaluator.extract(
        prompt=prompt_extract_stackoverflow_questions(),
        template_class=StackOverflowQuestionsData,
        extraction_name="stackoverflow_questions"
    )

    recommendations_data = await evaluator.extract(
        prompt=prompt_extract_recommendations(),
        template_class=RecommendationsData,
        extraction_name="recommendations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub pandas 'good first issue' search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F4:A1 - Switch to Issues tab
    mentions_github = has_any_ci(answer, ['github'])
    mentions_issues_tab = has_any_ci(answer, ['issues', 'issue tab', 'issues page'])
    evaluator.add_custom_node(
        result=bool(mentions_github and mentions_issues_tab),
        id="github_action_issues_tab",
        desc="[Action Node] github.com:F4:A1 - Navigate to GitHub and switch to the Issues tab",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A8 - Select sorting method (Recently updated)
    mentions_sorting = has_any_ci(answer, ['sort', 'sorted', 'recently updated', 'most active', 'active'])
    evaluator.add_custom_node(
        result=bool(mentions_sorting),
        id="github_action_sort_issues",
        desc="[Action Node] github.com:F4:A8 - Select issue sorting method (e.g., Recently updated)",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A4 - Browse through pages to find top 5
    mentions_pagination = has_any_ci(answer, ['page', 'next', 'browse', 'scroll', 'multiple'])
    has_multiple_issues = github_data and len(github_data.issues) >= 3
    evaluator.add_custom_node(
        result=bool(mentions_pagination or has_multiple_issues),
        id="github_action_pagination",
        desc="[Action Node] github.com:F1:A4 - Browse through multiple pages to compare issue activity",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F4:A20 - Click into issue details
    mentions_click_details = has_any_ci(answer, ['click', 'open', 'view', 'enter', 'detail'])
    has_summaries = github_data and any(issue.summary for issue in github_data.issues)
    evaluator.add_custom_node(
        result=bool(mentions_click_details or has_summaries),
        id="github_action_click_issue",
        desc="[Action Node] github.com:F4:A20 - Click into issue details to view description and comments",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F4:P2 - Understand issue list content
    has_titles = github_data and any(issue.title for issue in github_data.issues)
    has_discussion_counts = github_data and any(issue.discussion_count is not None for issue in github_data.issues)
    mentions_good_first_issue = has_any_ci(answer, ['good first issue', 'good-first-issue'])

    evaluator.add_custom_node(
        result=bool(has_titles and has_discussion_counts and mentions_good_first_issue),
        id="github_perception_issue_list",
        desc="[Perception Node] github.com:F4:P2 - Understand issue list items including labels, status, and activity metrics",
        parent=github_node,
        critical=False
    )

    # Check for top 5 issues with adequate information
    has_five_issues = github_data and len(github_data.issues) >= 5
    complete_issues = 0
    if github_data:
        for issue in github_data.issues:
            if issue.title and issue.summary and issue.discussion_count is not None:
                complete_issues += 1

    evaluator.add_custom_node(
        result=bool(has_five_issues and complete_issues >= 3),
        id="github_top5_completeness",
        desc="Provides information for top 5 most active issues (title, summary, discussion count)",
        parent=github_node,
        critical=False
    )

    # 3.2 StackOverflow section
    stackoverflow_node = evaluator.add_sequential(
        id="stackoverflow_section",
        desc="StackOverflow pandas questions sorted by votes",
        parent=root,
        critical=False
    )

    # [Action Node] stackoverflow.com:F2:A2 - Select sorting by votes
    mentions_stackoverflow = has_any_ci(answer, ['stackoverflow', 'stack overflow'])
    mentions_votes_sort = has_any_ci(answer, ['votes', 'voted', 'sort by votes', 'sorted by votes'])
    evaluator.add_custom_node(
        result=bool(mentions_stackoverflow and mentions_votes_sort),
        id="stackoverflow_action_sort_votes",
        desc="[Action Node] stackoverflow.com:F2:A2 - Select question list sorting method as 'Votes'",
        parent=stackoverflow_node,
        critical=False
    )

    # [Action Node] stackoverflow.com:F2:A9 - Browse pages for top 10
    so_pagination = has_any_ci(answer, ['page', 'next', 'browse', 'scroll'])
    has_multiple_questions = stackoverflow_data and len(stackoverflow_data.questions) >= 5
    evaluator.add_custom_node(
        result=bool(so_pagination or has_multiple_questions),
        id="stackoverflow_action_pagination",
        desc="[Action Node] stackoverflow.com:F2:A9 - Browse through pages to find top 10 questions",
        parent=stackoverflow_node,
        critical=False
    )

    # [Action Node] stackoverflow.com:F3:A29 - Hover over tags (optional)
    mentions_tags = has_any_ci(answer, ['tag', 'tagged', 'pandas tag'])
    evaluator.add_custom_node(
        result=bool(mentions_tags),
        id="stackoverflow_action_tag_preview",
        desc="[Action Node] stackoverflow.com:F3:A29 - Hover over tags to verify pandas relevance",
        parent=stackoverflow_node,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F2:P9 - Understand question list
    so_has_titles = stackoverflow_data and any(q.title for q in stackoverflow_data.questions)
    so_has_votes = stackoverflow_data and any(q.vote_count is not None for q in stackoverflow_data.questions)
    mentions_pandas = has_any_ci(answer, ['pandas'])

    evaluator.add_custom_node(
        result=bool(so_has_titles and so_has_votes and mentions_pandas),
        id="stackoverflow_perception_question_list",
        desc="[Perception Node] stackoverflow.com:F2:P9 - Understand question list topics and popularity metrics",
        parent=stackoverflow_node,
        critical=False
    )

    # Check for top 10 questions with adequate information
    has_ten_questions = stackoverflow_data and len(stackoverflow_data.questions) >= 10
    complete_questions = 0
    if stackoverflow_data:
        for question in stackoverflow_data.questions:
            if question.title and question.vote_count is not None:
                complete_questions += 1

    evaluator.add_custom_node(
        result=bool(has_ten_questions and complete_questions >= 7),
        id="stackoverflow_top10_completeness",
        desc="Provides titles and vote counts for top 10 pandas-related questions",
        parent=stackoverflow_node,
        critical=False
    )

    # 3.3 Cross-comparison and recommendation section
    recommendation_node = evaluator.add_sequential(
        id="recommendation_section",
        desc="Cross-comparison of StackOverflow questions with GitHub issues and recommendations",
        parent=root,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F4:P21 - Understand content relationships
    comparison_mentioned = has_any_ci(answer, ['compare', 'comparison', 'cross', 'match', 'relate', 'align', 'correspond'])
    has_comparison = recommendations_data and recommendations_data.comparison_performed
    evaluator.add_custom_node(
        result=bool(comparison_mentioned or has_comparison),
        id="stackoverflow_perception_relationship",
        desc="[Perception Node] stackoverflow.com:F4:P21 - Understand thematic relationships between questions",
        parent=recommendation_node,
        critical=False
    )

    # Check for meaningful recommendations
    has_recommendations = recommendations_data and len(recommendations_data.recommended_issues) >= 2
    mentions_beginner_suitable = has_any_ci(answer, ['beginner', 'suitable', 'worthwhile', 'recommend', 'pain point', 'user need'])

    evaluator.add_custom_node(
        result=bool(has_recommendations and mentions_beginner_suitable),
        id="recommendation_quality",
        desc="Provides 2-3 recommendations with justification based on beginner-friendliness and user pain points",
        parent=recommendation_node,
        critical=False
    )

    # Overall integration check
    both_sources_present = (github_data and len(github_data.issues) >= 3 and
                           stackoverflow_data and len(stackoverflow_data.questions) >= 5)
    evaluator.add_custom_node(
        result=bool(both_sources_present and has_recommendations),
        id="integration_completeness",
        desc="Successfully integrates data from both GitHub and StackOverflow to make recommendations",
        parent=recommendation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
