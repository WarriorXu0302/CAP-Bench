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
TASK_ID = "task-bf5cee"
TASK_DESCRIPTION = 'I am creating a 2025 annual review video focusing on the differences between The Last of Us game and its TV adaptation. Please help me gather core materials:\n\nFirst, go to **IGN** and search for "The Last of Us Part I". Find the review article specifically for the **PS5 version**. Confirm if this review received an "Editor\'s Choice" tag (or a score of 9 or higher), and note down the reviewer\'s name.\n\nNext, go to **IMDb** and search for "The Last of Us" (TV Series). In the cast and crew list, locate the actor who plays "Joel". Click on their profile page and check the "Known For" section, then record the title of their top-listed representative work. Simultaneously, return to the series\' main page, expand the "Storyline" section to view "Taglines", and record the first tagline listed.\n\nFinally, go to **Metacritic** and search for "The Last of Us Part I" (PlayStation 5) and "The Last of Us" (TV Show Season 1) separately. We need to compare the audience reception for both, so please record the **User Score** for each.\n\n**Output:** IGN Score, IGN Reviewer Name, Whether \'Editor\'s Choice\' was awarded, Name of the actor playing Joel, Actor\'s top representative work, First tagline of the TV series, Metacritic Game User Score, Metacritic TV Series User Score, and the detailed page links used for each step.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IGNReviewInfo(BaseModel):
    """IGN review information extracted from the answer"""
    ign_score: Optional[str] = None
    reviewer_name: Optional[str] = None
    editors_choice: Optional[bool] = None
    ign_review_url: Optional[str] = None


class IMDbActorInfo(BaseModel):
    """IMDb actor and tagline information extracted from the answer"""
    joel_actor_name: Optional[str] = None
    actor_known_for_first: Optional[str] = None
    series_first_tagline: Optional[str] = None
    imdb_actor_url: Optional[str] = None
    imdb_series_url: Optional[str] = None


class MetacriticScores(BaseModel):
    """Metacritic user scores extracted from the answer"""
    game_user_score: Optional[str] = None
    tv_user_score: Optional[str] = None
    metacritic_game_url: Optional[str] = None
    metacritic_tv_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ign_info() -> str:
    return """
Extract the IGN review information for The Last of Us Part I (PS5 version) from the answer.

Return:
- ign_score: the review score exactly as stated (e.g., "9/10", "9.0", "10").
- reviewer_name: the name of the reviewer who wrote the review.
- editors_choice: true if the review received an "Editor's Choice" tag or award, false otherwise. If not mentioned, set null.
- ign_review_url: the URL of the IGN review page if provided.

If any field is missing, set it to null.
"""


def prompt_extract_imdb_info() -> str:
    return """
Extract the IMDb information about The Last of Us TV series from the answer.

Return:
- joel_actor_name: the name of the actor who plays Joel in the TV series.
- actor_known_for_first: the first/top item listed in the "Known For" section on the actor's IMDb profile page.
- series_first_tagline: the first tagline listed under the "Taglines" section of The Last of Us TV series main page.
- imdb_actor_url: the URL of the actor's IMDb profile page if provided.
- imdb_series_url: the URL of the TV series main page if provided.

If any field is missing, set it to null.
"""


def prompt_extract_metacritic_scores() -> str:
    return """
Extract the Metacritic User Score information from the answer.

Return:
- game_user_score: the User Score for The Last of Us Part I (PlayStation 5) on Metacritic.
- tv_user_score: the User Score for The Last of Us (TV Show Season 1) on Metacritic.
- metacritic_game_url: the URL of the game's Metacritic page if provided.
- metacritic_tv_url: the URL of the TV show's Metacritic page if provided.

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


def extract_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_score(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'www.', '.com', '.org'])


def looks_like_user_score(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # User scores typically range 0-10
    return 0 <= num <= 10


def mentions_ps5_version(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['ps5', 'playstation 5'])


def mentions_editors_choice(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ["editor's choice", "editors choice", "editor choice"])


def mentions_known_for(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['known for'])


def mentions_tagline(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['tagline'])


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
    ign_info = await evaluator.extract(
        prompt=prompt_extract_ign_info(),
        template_class=IGNReviewInfo,
        extraction_name="ign_review_info"
    )

    imdb_info = await evaluator.extract(
        prompt=prompt_extract_imdb_info(),
        template_class=IMDbActorInfo,
        extraction_name="imdb_actor_info"
    )

    metacritic_info = await evaluator.extract(
        prompt=prompt_extract_metacritic_scores(),
        template_class=MetacriticScores,
        extraction_name="metacritic_scores"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IGN section
    ign_node = evaluator.add_sequential(
        id="ign_section",
        desc="IGN review for The Last of Us Part I (PS5 version)",
        parent=root,
        critical=False
    )

    # [Action Node] ign.com:F2:A7 - Navigate to IGN and find PS5 review
    ign_navigation_ok = (has_any_ci(answer, ['ign']) and
                         has_any_ci(answer, ['the last of us', 'last of us part i', 'tlou']))
    evaluator.add_custom_node(
        result=bool(ign_navigation_ok),
        id="ign_action_navigate",
        desc="[Action Node] ign.com:F2:A7 - Navigate to IGN and search for The Last of Us Part I review",
        parent=ign_node,
        critical=False
    )

    # [Action Node] ign.com:F2:A5 - Filter for PS5 version specifically
    ps5_filtering_ok = mentions_ps5_version(answer) and looks_like_url(ign_info.ign_review_url)
    evaluator.add_custom_node(
        result=bool(ps5_filtering_ok),
        id="ign_action_filter_ps5",
        desc="[Action Node] ign.com:F2:A5 - Filter and select the PS5 version review from search results",
        parent=ign_node,
        critical=False
    )

    # [Perception Node] ign.com:F2:P1 - Identify Editor's Choice status
    editors_choice_detected = (ign_info.editors_choice is not None) or mentions_editors_choice(answer)
    evaluator.add_custom_node(
        result=bool(editors_choice_detected),
        id="ign_perception_editors_choice",
        desc="[Perception Node] ign.com:F2:P1 - Identify whether the review received Editor's Choice tag",
        parent=ign_node,
        critical=False
    )

    # Extract IGN score and reviewer name (non-prefixed)
    ign_score_ok = looks_like_score(ign_info.ign_score)
    evaluator.add_custom_node(
        result=bool(ign_score_ok),
        id="ign_extract_score",
        desc="Extract the IGN review score",
        parent=ign_node,
        critical=False
    )

    ign_reviewer_ok = bool(ign_info.reviewer_name and ign_info.reviewer_name.strip())
    evaluator.add_custom_node(
        result=bool(ign_reviewer_ok),
        id="ign_extract_reviewer",
        desc="Extract the reviewer's name from the IGN review",
        parent=ign_node,
        critical=False
    )

    # 3.2 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb information for The Last of Us TV series and Joel's actor",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F3:A39 - Navigate to actor profile
    imdb_actor_navigation_ok = (has_any_ci(answer, ['imdb']) and
                                 has_any_ci(answer, ['joel']) and
                                 bool(imdb_info.joel_actor_name and imdb_info.joel_actor_name.strip()))
    evaluator.add_custom_node(
        result=bool(imdb_actor_navigation_ok),
        id="imdb_action_actor_profile",
        desc="[Action Node] imdb.com:F3:A39 - Locate Joel's actor in cast list and navigate to their profile page",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F3:P17 - Understand role-to-actor mapping
    joel_actor_understood = bool(imdb_info.joel_actor_name and imdb_info.joel_actor_name.strip())
    evaluator.add_custom_node(
        result=bool(joel_actor_understood),
        id="imdb_perception_joel_actor",
        desc="[Perception Node] imdb.com:F3:P17 - Understand the mapping between Joel character and the actor",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A28 - Identify and extract from Known For section
    known_for_extraction_ok = (bool(imdb_info.actor_known_for_first and imdb_info.actor_known_for_first.strip()) and
                                mentions_known_for(answer))
    evaluator.add_custom_node(
        result=bool(known_for_extraction_ok),
        id="imdb_action_known_for",
        desc="[Action Node] imdb.com:F3:A28 - Locate the Known For section and extract the first/top item",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A34 - Expand Storyline/Taglines section
    tagline_expansion_ok = (bool(imdb_info.series_first_tagline and imdb_info.series_first_tagline.strip()) and
                            mentions_tagline(answer))
    evaluator.add_custom_node(
        result=bool(tagline_expansion_ok),
        id="imdb_action_expand_tagline",
        desc="[Action Node] imdb.com:F3:A34 - Expand the Storyline section to view Taglines and extract the first one",
        parent=imdb_node,
        critical=False
    )

    # 3.3 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic User Scores for both game and TV show",
        parent=root,
        critical=False
    )

    # [Action Node] metacritic.com:F4:A6 - Switch to User Score tab/view
    metacritic_tab_switch_ok = (has_any_ci(answer, ['metacritic']) and
                                 has_any_ci(answer, ['user score']))
    evaluator.add_custom_node(
        result=bool(metacritic_tab_switch_ok),
        id="metacritic_action_user_score_tab",
        desc="[Action Node] metacritic.com:F4:A6 - Navigate to User Score section/tab (not Metascore)",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.com:F4:P1 - Extract game user score
    game_user_score_ok = looks_like_user_score(metacritic_info.game_user_score)
    evaluator.add_custom_node(
        result=bool(game_user_score_ok),
        id="metacritic_perception_game_score",
        desc="[Perception Node] metacritic.com:F4:P1 - Extract User Score for The Last of Us Part I (game)",
        parent=metacritic_node,
        critical=False
    )

    # Extract TV user score (non-prefixed but tied to same perception ability)
    tv_user_score_ok = looks_like_user_score(metacritic_info.tv_user_score)
    evaluator.add_custom_node(
        result=bool(tv_user_score_ok),
        id="metacritic_extract_tv_score",
        desc="Extract User Score for The Last of Us (TV show)",
        parent=metacritic_node,
        critical=False
    )

    # Verify both scores are present for comparison
    comparison_ok = game_user_score_ok and tv_user_score_ok
    evaluator.add_custom_node(
        result=bool(comparison_ok),
        id="metacritic_comparison_ready",
        desc="Both game and TV User Scores are available for comparison",
        parent=metacritic_node,
        critical=False
    )

    # 3.4 Overall completeness checks (non-prefixed)
    all_urls_present = (looks_like_url(ign_info.ign_review_url) and
                        looks_like_url(imdb_info.imdb_actor_url) and
                        looks_like_url(imdb_info.imdb_series_url) and
                        looks_like_url(metacritic_info.metacritic_game_url) and
                        looks_like_url(metacritic_info.metacritic_tv_url))
    evaluator.add_custom_node(
        result=bool(all_urls_present),
        id="all_urls_provided",
        desc="All required page URLs are provided in the answer",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
