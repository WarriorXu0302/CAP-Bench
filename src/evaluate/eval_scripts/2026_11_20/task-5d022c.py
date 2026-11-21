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
TASK_ID = "task-5d022c"
TASK_DESCRIPTION = "I am developing a pixel-art action Roguelike game and would like to research the map generation mechanisms of benchmark products.\n\nFirst, go to Steam. Use the filtering function to find games tagged with both 'Pixel Graphics' and 'Action Roguelike'. To ensure quality, sort the results by 'User Reviews', and identify the top-ranked game that is paid (not free). Go to its detail page and confirm if its positive review percentage exceeds 95%.\n\nOnce confirmed, I want to find similar open-source implementations. Go to GitHub and search for 'Roguelike dungeon generation', filter the language to C#, sort by 'Stars', and identify the top-ranked repository.\n\nFinally, compile for me the name and positive review percentage of the Steam game, as well as the URL and star count of the GitHub repository."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SteamGameInfo(BaseModel):
    """Steam game information extracted from the answer"""
    game_name: Optional[str] = None
    positive_review_percentage: Optional[str] = None


class GitHubRepoInfo(BaseModel):
    """GitHub repository information extracted from the answer"""
    repo_url: Optional[str] = None
    star_count: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_steam_game_from_answer() -> str:
    return """
Extract the Steam game information from the answer:

- game_name: the name of the top-ranked paid game found with both 'Pixel Graphics' and 'Action Roguelike' tags, sorted by 'User Reviews'.
- positive_review_percentage: the positive review percentage of this game exactly as stated (include % symbol if present).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_github_repo_from_answer() -> str:
    return """
Extract the GitHub repository information from the answer:

- repo_url: the URL of the top-ranked repository found when searching for 'Roguelike dungeon generation' with language filtered to C# and sorted by Stars.
- star_count: the star count of this repository exactly as stated (include any formatting if present).

If any field is missing in the answer, set it to null.
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


def extract_percentage(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)\s*%?', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def extract_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Remove commas and extract number
    cleaned = re.sub(r'[,\s]', '', text)
    m = re.search(r'(\d+)', cleaned)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def looks_like_github_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'github.com')


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
    steam_info = await evaluator.extract(
        prompt=prompt_extract_steam_game_from_answer(),
        template_class=SteamGameInfo,
        extraction_name="steam_game_info"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_repo_from_answer(),
        template_class=GitHubRepoInfo,
        extraction_name="github_repo_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Steam part
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam game research with multi-tag filtering and sorting",
        parent=root,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A1 - Multi-tag filtering
    steam_multi_tag_ok = (has_any_ci(answer, ['steam']) and
                          has_any_ci(answer, ['pixel graphics']) and
                          has_any_ci(answer, ['action roguelike']))
    evaluator.add_custom_node(
        result=bool(steam_multi_tag_ok),
        id="steam_action_multi_tag",
        desc="[Action Node] store.steampowered.com:F1:A1 - Apply multi-tag filtering for both 'Pixel Graphics' and 'Action Roguelike'",
        parent=steam_node,
        critical=False
    )

    # [Action Node] store.steampowered.com:F1:A4 - Sort by User Reviews
    steam_sort_ok = has_any_ci(answer, ['user reviews', 'sort', 'sorted'])
    evaluator.add_custom_node(
        result=bool(steam_sort_ok),
        id="steam_action_sort",
        desc="[Action Node] store.steampowered.com:F1:A4 - Sort results by 'User Reviews'",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F2:P15 - Verify positive review percentage exceeds 95%
    percentage_value = extract_percentage(steam_info.positive_review_percentage)
    percentage_ok = percentage_value is not None and percentage_value > 95
    percentage_mentioned = has_any_ci(answer, ['95%', '95 %', 'positive', 'review'])

    evaluator.add_custom_node(
        result=bool(percentage_ok and percentage_mentioned),
        id="steam_perception_review_percentage",
        desc="[Perception Node] store.steampowered.com:F2:P15 - Confirm positive review percentage exceeds 95%",
        parent=steam_node,
        critical=False
    )

    # Non-prefixed checks for Steam
    game_name_ok = bool(steam_info.game_name and steam_info.game_name.strip())
    evaluator.add_custom_node(
        result=bool(game_name_ok),
        id="steam_has_game_name",
        desc="Provides the name of the identified Steam game",
        parent=steam_node,
        critical=False
    )

    paid_mention_ok = has_any_ci(answer, ['paid', 'not free', 'purchase'])
    evaluator.add_custom_node(
        result=bool(paid_mention_ok),
        id="steam_mentions_paid",
        desc="Mentions identifying a paid (not free) game",
        parent=steam_node,
        critical=False
    )

    top_ranked_ok = has_any_ci(answer, ['top', 'first', 'highest', 'rank'])
    evaluator.add_custom_node(
        result=bool(top_ranked_ok),
        id="steam_mentions_top_ranked",
        desc="Mentions identifying the top-ranked or first result",
        parent=steam_node,
        critical=False
    )

    # 3.2 GitHub part
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository search with language and sorting filters",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A6 - Language filter to C#
    github_language_ok = (has_any_ci(answer, ['github']) and
                          has_any_ci(answer, ['c#', 'csharp', 'c sharp']))
    evaluator.add_custom_node(
        result=bool(github_language_ok),
        id="github_action_language_filter",
        desc="[Action Node] github.com:F1:A6 - Filter search results by language (C#)",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by Stars
    github_sort_ok = has_any_ci(answer, ['stars', 'star', 'sort'])
    evaluator.add_custom_node(
        result=bool(github_sort_ok),
        id="github_action_sort_stars",
        desc="[Action Node] github.com:F1:A7 - Sort repositories by Star count",
        parent=github_node,
        critical=False
    )

    # Non-prefixed checks for GitHub
    search_query_ok = has_any_ci(answer, ['roguelike dungeon generation', 'dungeon generation', 'roguelike'])
    evaluator.add_custom_node(
        result=bool(search_query_ok),
        id="github_mentions_search_query",
        desc="Mentions searching for 'Roguelike dungeon generation' or similar terms",
        parent=github_node,
        critical=False
    )

    repo_url_ok = looks_like_github_url(github_info.repo_url)
    evaluator.add_custom_node(
        result=bool(repo_url_ok),
        id="github_has_repo_url",
        desc="Provides a valid GitHub repository URL",
        parent=github_node,
        critical=False
    )

    star_count_value = extract_number(github_info.star_count)
    star_count_ok = star_count_value is not None
    evaluator.add_custom_node(
        result=bool(star_count_ok),
        id="github_has_star_count",
        desc="Provides the star count of the repository",
        parent=github_node,
        critical=False
    )

    top_repo_ok = has_any_ci(answer, ['top', 'first', 'highest'])
    evaluator.add_custom_node(
        result=bool(top_repo_ok),
        id="github_mentions_top_ranked",
        desc="Mentions identifying the top-ranked repository",
        parent=github_node,
        critical=False
    )

    # 3.3 Final compilation check
    compilation_node = evaluator.add_parallel(
        id="compilation_section",
        desc="Final compilation of all requested information",
        parent=root,
        critical=False
    )

    all_steam_data = game_name_ok and (steam_info.positive_review_percentage is not None)
    evaluator.add_custom_node(
        result=bool(all_steam_data),
        id="compilation_steam_complete",
        desc="Compiles Steam game name and positive review percentage",
        parent=compilation_node,
        critical=False
    )

    all_github_data = repo_url_ok and star_count_ok
    evaluator.add_custom_node(
        result=bool(all_github_data),
        id="compilation_github_complete",
        desc="Compiles GitHub repository URL and star count",
        parent=compilation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
