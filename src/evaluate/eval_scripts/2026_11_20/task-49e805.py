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
TASK_ID = "task-49e805"
TASK_DESCRIPTION = "As a Cyberpunk 2077 player, I've encountered an issue where a recent automatic game update has rendered my core mod, 'Cyber Engine Tweaks', inoperable. Could you please assist me in performing a compatibility troubleshooting process:\n\n1.  First, navigate to the 'News' section on the Cyberpunk 2077 Steam store page, find, and record the game's latest version number (Patch number).\n2.  Then, search GitHub for 'Cyber Engine Tweaks' and proceed to the repository with the most stars.\n3.  On the 'Releases' page, check if the latest published Mod version explicitly states support for the Steam game version you just found.\n4.  Next, a crucial step: go to the 'Issues' page, sort the issues by 'Recently updated' using the filter, and examine the first few 'Open' issues to determine if there are widespread user reports of crashes for this version.\n\nFinally, please provide me with:\n*   The current game version number.\n*   The recommended Mod version to download.\n*   Whether it is advisable to update the Mod at this time, based on the Issues review."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameVersion(BaseModel):
    """Game version extracted from the answer"""
    version_number: Optional[str] = None
    patch_number: Optional[str] = None


class ModInformation(BaseModel):
    """Mod information extracted from the answer"""
    mod_name: Optional[str] = None
    repository_mention: Optional[str] = None
    mod_version: Optional[str] = None
    supported_game_version: Optional[str] = None


class IssuesAnalysis(BaseModel):
    """Issues analysis extracted from the answer"""
    issues_checked: Optional[bool] = None
    crash_reports_mentioned: Optional[bool] = None
    stability_assessment: Optional[str] = None


class Recommendation(BaseModel):
    """Final recommendation extracted from the answer"""
    should_update: Optional[str] = None
    reasoning: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_game_version() -> str:
    return """
Extract the Cyberpunk 2077 game version information from the answer that was found on the Steam store page News section.

Return:
- version_number: the game version or patch number exactly as stated (e.g., "2.2", "Patch 2.2", "2.13"). If not present, set null.
- patch_number: if explicitly mentioned as a patch number, extract it here. Otherwise, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_mod_info() -> str:
    return """
From the answer, extract information about the 'Cyber Engine Tweaks' mod found on GitHub:

- mod_name: the name of the mod as mentioned.
- repository_mention: any mention of the GitHub repository (e.g., "most starred", "top result").
- mod_version: the latest mod version number found on the Releases page.
- supported_game_version: the game version that the mod explicitly supports according to the Releases page.

If any field is missing, set it to null.
"""


def prompt_extract_issues_analysis() -> str:
    return """
From the answer, extract the analysis of GitHub Issues:

- issues_checked: true if the answer mentions checking the Issues page, false otherwise.
- crash_reports_mentioned: true if crash reports or stability issues are discussed, false otherwise.
- stability_assessment: any assessment of stability based on recent issues (e.g., "widespread crashes", "stable", "issues reported").

If any field is missing, set it to null.
"""


def prompt_extract_recommendation() -> str:
    return """
From the answer, extract the final recommendation:

- should_update: whether it is advisable to update the mod ("yes", "no", "cautious", etc.).
- reasoning: the reasoning behind the recommendation.

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


def looks_like_version(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept version patterns like "2.2", "2.13", "Patch 2.2", "v2.2", "1.63"
    patterns = [
        r'\d+\.\d+',
        r'patch\s+\d+\.\d+',
        r'v\d+\.\d+',
        r'version\s+\d+\.\d+'
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def extract_version_number(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r'(\d+\.\d+(?:\.\d+)?)', text)
    if m:
        return m.group(1)
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
    game_version_info = await evaluator.extract(
        prompt=prompt_extract_game_version(),
        template_class=GameVersion,
        extraction_name="game_version"
    )

    mod_info = await evaluator.extract(
        prompt=prompt_extract_mod_info(),
        template_class=ModInformation,
        extraction_name="mod_information"
    )

    issues_analysis = await evaluator.extract(
        prompt=prompt_extract_issues_analysis(),
        template_class=IssuesAnalysis,
        extraction_name="issues_analysis"
    )

    recommendation_info = await evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=Recommendation,
        extraction_name="recommendation"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Steam News section - Get game version
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam store page - Cyberpunk 2077 News section for game version",
        parent=root,
        critical=False
    )

    steam_navigation_ok = (has_any_ci(answer, ['steam']) and
                          has_any_ci(answer, ['cyberpunk 2077', 'cyberpunk']) and
                          has_any_ci(answer, ['news']))
    evaluator.add_custom_node(
        result=bool(steam_navigation_ok),
        id="steam_navigation",
        desc="Navigate to the Steam store page News section for Cyberpunk 2077",
        parent=steam_node,
        critical=False
    )

    game_version_extracted = (looks_like_version(game_version_info.version_number) or
                             looks_like_version(game_version_info.patch_number))
    evaluator.add_custom_node(
        result=bool(game_version_extracted),
        id="steam_game_version_extraction",
        desc="Extract the current game version/patch number from Steam News",
        parent=steam_node,
        critical=False
    )

    # 3.2 GitHub search and repository
    github_search_node = evaluator.add_sequential(
        id="github_search_section",
        desc="GitHub - Search for Cyber Engine Tweaks repository",
        parent=root,
        critical=False
    )

    github_search_ok = (has_any_ci(answer, ['github']) and
                       has_any_ci(answer, ['cyber engine tweaks']))
    evaluator.add_custom_node(
        result=bool(github_search_ok),
        id="github_search",
        desc="Search GitHub for 'Cyber Engine Tweaks'",
        parent=github_search_node,
        critical=False
    )

    most_starred_mentioned = has_any_ci(answer, ['most stars', 'most starred', 'highest stars', 'top starred'])
    evaluator.add_custom_node(
        result=bool(most_starred_mentioned),
        id="github_most_starred",
        desc="Identify and select the repository with the most stars",
        parent=github_search_node,
        critical=False
    )

    # 3.3 GitHub Releases - Check compatibility
    github_releases_node = evaluator.add_sequential(
        id="github_releases_section",
        desc="GitHub Releases page - Check mod version and game compatibility",
        parent=root,
        critical=False
    )

    releases_navigation_ok = has_any_ci(answer, ['releases', 'release page'])
    evaluator.add_custom_node(
        result=bool(releases_navigation_ok),
        id="github_releases_navigation",
        desc="Navigate to the Releases page",
        parent=github_releases_node,
        critical=False
    )

    # [Perception Node] github.com:F6:P4 - Version understanding
    mod_version_ok = looks_like_version(mod_info.mod_version)
    supported_version_ok = looks_like_version(mod_info.supported_game_version)
    compatibility_check_ok = mod_version_ok and supported_version_ok

    evaluator.add_custom_node(
        result=bool(compatibility_check_ok),
        id="github_releases_compatibility",
        desc="[Perception Node] github.com:F6:P4 - Check if the latest Mod version explicitly states support for the Steam game version",
        parent=github_releases_node,
        critical=False
    )

    # 3.4 GitHub Issues - Stability check
    github_issues_node = evaluator.add_sequential(
        id="github_issues_section",
        desc="GitHub Issues page - Check for crash reports and stability",
        parent=root,
        critical=False
    )

    issues_navigation_ok = has_any_ci(answer, ['issues', 'issue page'])
    evaluator.add_custom_node(
        result=bool(issues_navigation_ok),
        id="github_issues_navigation",
        desc="Navigate to the Issues page",
        parent=github_issues_node,
        critical=False
    )

    # [Action Node] github.com:F4:A8 - Sort by Recently updated
    sort_by_recent_ok = has_any_ci(answer, ['recently updated', 'sort', 'sorted by', 'filter'])
    evaluator.add_custom_node(
        result=bool(sort_by_recent_ok),
        id="github_issues_sort",
        desc="[Action Node] github.com:F4:A8 - Sort issues by 'Recently updated' using the filter",
        parent=github_issues_node,
        critical=False
    )

    # [Perception Node] github.com:F4:P2 - List content understanding
    open_issues_checked = has_any_ci(answer, ['open', 'open issues'])
    crash_check_ok = (has_any_ci(answer, ['crash', 'crashes', 'crashing', 'stability', 'bug', 'issue']) and
                     open_issues_checked)

    evaluator.add_custom_node(
        result=bool(crash_check_ok),
        id="github_issues_crash_check",
        desc="[Perception Node] github.com:F4:P2 - Examine first few Open issues to check for widespread crash reports",
        parent=github_issues_node,
        critical=False
    )

    # 3.5 Final outputs required
    final_outputs_node = evaluator.add_parallel(
        id="final_outputs",
        desc="Final required outputs for the user",
        parent=root,
        critical=False
    )

    game_version_provided = game_version_extracted
    evaluator.add_custom_node(
        result=bool(game_version_provided),
        id="output_game_version",
        desc="Provide the current game version number",
        parent=final_outputs_node,
        critical=False
    )

    mod_version_provided = mod_version_ok
    evaluator.add_custom_node(
        result=bool(mod_version_provided),
        id="output_mod_version",
        desc="Provide the recommended Mod version to download",
        parent=final_outputs_node,
        critical=False
    )

    recommendation_provided = bool(recommendation_info.should_update and recommendation_info.should_update.strip())
    evaluator.add_custom_node(
        result=bool(recommendation_provided),
        id="output_recommendation",
        desc="Provide recommendation on whether to update the Mod based on Issues review",
        parent=final_outputs_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
