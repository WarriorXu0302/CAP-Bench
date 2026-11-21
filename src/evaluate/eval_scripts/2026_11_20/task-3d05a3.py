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
TASK_ID = "task-3d05a3"
TASK_DESCRIPTION = "I just listened to Kanye West's 'Bound 2' and particularly loved the high-pitched vocal sample in it.\nPlease go to YouTube and find this song. Check the comments section to see if anyone mentions the name of the original song from which the sample was taken (remember to scroll through comments or sort by top comments).\nOnce the original song's name is identified, search for it on Genius.com. On the lyrics page, examine the annotations (by clicking on the highlighted lyrics) to confirm which specific lyric line was sampled in 'Bound 2'.\nFinally, I want to purchase a vinyl record of this original song for my collection. Go to Discogs.com, search for the album containing this original song, filter the format by 'Vinyl', and find an available item with a 'Media Condition' rating of 'Near Mint' or 'Mint' and priced under $50. Provide me with the purchase link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OriginalSongInfo(BaseModel):
    """Original song information extracted from the answer"""
    original_song_name: Optional[str] = None
    original_artist: Optional[str] = None


class SampledLyricInfo(BaseModel):
    """Sampled lyric information extracted from the answer"""
    sampled_lyric_line: Optional[str] = None


class VinylPurchaseInfo(BaseModel):
    """Vinyl purchase information extracted from the answer"""
    album_name: Optional[str] = None
    media_condition: Optional[str] = None
    price_text: Optional[str] = None
    purchase_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_original_song() -> str:
    return """
Extract the original song information that was sampled in Kanye West's 'Bound 2' from the answer.

Return:
- original_song_name: the name of the original song exactly as mentioned in the answer.
- original_artist: the artist of the original song if mentioned.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_sampled_lyric() -> str:
    return """
From the answer, extract the specific lyric line from the original song that was sampled in 'Bound 2'.

Return:
- sampled_lyric_line: the exact lyric line text that was sampled, as reported in the answer.

If not present, set it to null.
"""


def prompt_extract_vinyl_purchase() -> str:
    return """
From the answer, extract the vinyl purchase information for the original sampled song.

Return:
- album_name: the album name containing the original song.
- media_condition: the media condition rating (e.g., "Near Mint", "Mint", "NM", "M").
- price_text: the price exactly as stated (include currency symbol if present).
- purchase_link: the purchase link or URL provided.

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


def contains_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://|www\.', text, re.IGNORECASE))


def mentions_sorting_or_scrolling(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['sort', 'sorted', 'scroll', 'scrolled', 'top comments', 'top comment'])


def mentions_clicking_or_annotation(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['click', 'clicked', 'annotation', 'annotated', 'highlight', 'highlighted'])


def mentions_vinyl_filter(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['vinyl', 'format'])


def mentions_condition_filter(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['near mint', 'mint', 'nm', 'media condition', 'condition'])


def extract_price_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d{2})?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def is_valid_condition(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['near mint', 'mint', 'nm', 'm'])


def is_discogs_link(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'discogs.com')


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
    original_song_info = await evaluator.extract(
        prompt=prompt_extract_original_song(),
        template_class=OriginalSongInfo,
        extraction_name="original_song_info"
    )

    sampled_lyric_info = await evaluator.extract(
        prompt=prompt_extract_sampled_lyric(),
        template_class=SampledLyricInfo,
        extraction_name="sampled_lyric_info"
    )

    vinyl_info = await evaluator.extract(
        prompt=prompt_extract_vinyl_purchase(),
        template_class=VinylPurchaseInfo,
        extraction_name="vinyl_purchase_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube - Find 'Bound 2' and identify original song from comments",
        parent=root,
        critical=False
    )

    # Check if answer mentions YouTube and Bound 2
    youtube_action_ok = has_any_ci(answer, ['youtube']) and has_any_ci(answer, ['bound 2', 'bound2'])
    evaluator.add_custom_node(
        result=bool(youtube_action_ok),
        id="youtube_action_find_song",
        desc="Navigate to YouTube and find 'Bound 2' by Kanye West",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F5:A60 - Sort by top comments (dropdown menu operation)
    sort_action_ok = mentions_sorting_or_scrolling(answer) and has_any_ci(answer, ['comment', 'comments'])
    evaluator.add_custom_node(
        result=bool(sort_action_ok),
        id="youtube_action_sort_comments",
        desc="[Action Node] youtube.com:F5:A60 - Sort comments by top comments using dropdown menu",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F5:A28 - Scroll through comments
    scroll_action_ok = has_any_ci(answer, ['scroll', 'scrolled']) and has_any_ci(answer, ['comment', 'comments'])
    evaluator.add_custom_node(
        result=bool(scroll_action_ok),
        id="youtube_action_scroll_comments",
        desc="[Action Node] youtube.com:F5:A28 - Scroll through comments to load more",
        parent=youtube_node,
        critical=False
    )

    # Check if original song was identified
    original_song_identified = bool(original_song_info and original_song_info.original_song_name and original_song_info.original_song_name.strip())
    evaluator.add_custom_node(
        result=bool(original_song_identified),
        id="youtube_perception_original_song",
        desc="Identify the original song name from YouTube comments",
        parent=youtube_node,
        critical=False
    )

    # 3.2 Genius section
    genius_node = evaluator.add_sequential(
        id="genius_section",
        desc="Genius.com - Search original song and examine annotations",
        parent=root,
        critical=False
    )

    # Check if answer mentions Genius
    genius_action_ok = has_any_ci(answer, ['genius', 'genius.com'])
    evaluator.add_custom_node(
        result=bool(genius_action_ok),
        id="genius_action_search",
        desc="Navigate to Genius.com and search for the original song",
        parent=genius_node,
        critical=False
    )

    # [Action Node] genius.com:F3:A12 - Click highlighted lyrics to trigger annotation bubble
    click_annotation_ok = mentions_clicking_or_annotation(answer) and has_any_ci(answer, ['lyric', 'lyrics'])
    evaluator.add_custom_node(
        result=bool(click_annotation_ok),
        id="genius_action_click_annotation",
        desc="[Action Node] genius.com:F3:A12 - Click on highlighted lyrics to view annotations",
        parent=genius_node,
        critical=False
    )

    # Check if sampled lyric line was identified
    sampled_lyric_identified = bool(sampled_lyric_info and sampled_lyric_info.sampled_lyric_line and sampled_lyric_info.sampled_lyric_line.strip())
    evaluator.add_custom_node(
        result=bool(sampled_lyric_identified),
        id="genius_perception_sampled_lyric",
        desc="Identify the specific lyric line that was sampled in 'Bound 2'",
        parent=genius_node,
        critical=False
    )

    # 3.3 Discogs section
    discogs_node = evaluator.add_sequential(
        id="discogs_section",
        desc="Discogs.com - Search album, filter vinyl, find Near Mint/Mint item under $50",
        parent=root,
        critical=False
    )

    # Check if answer mentions Discogs
    discogs_action_ok = has_any_ci(answer, ['discogs', 'discogs.com'])
    evaluator.add_custom_node(
        result=bool(discogs_action_ok),
        id="discogs_action_search",
        desc="Navigate to Discogs.com and search for the album",
        parent=discogs_node,
        critical=False
    )

    # [Action Node] discogs.com:F4:A2 - Filter by Vinyl format (multi-select operation)
    vinyl_filter_ok = mentions_vinyl_filter(answer)
    evaluator.add_custom_node(
        result=bool(vinyl_filter_ok),
        id="discogs_action_filter_vinyl",
        desc="[Action Node] discogs.com:F4:A2 - Filter format by 'Vinyl' using multi-select",
        parent=discogs_node,
        critical=False
    )

    # [Action Node] discogs.com:F4:A2 - Filter by condition (Near Mint or Mint)
    condition_filter_ok = mentions_condition_filter(answer)
    evaluator.add_custom_node(
        result=bool(condition_filter_ok),
        id="discogs_action_filter_condition",
        desc="[Action Node] discogs.com:F4:A2 - Filter by Media Condition (Near Mint or Mint)",
        parent=discogs_node,
        critical=False
    )

    # [Perception Node] discogs.com:F4:P5 - Identify Media Condition label on item card
    condition_value_ok = is_valid_condition(vinyl_info.media_condition)
    evaluator.add_custom_node(
        result=bool(condition_value_ok),
        id="discogs_perception_condition_label",
        desc="[Perception Node] discogs.com:F4:P5 - Identify Media Condition label (NM/M) on item card",
        parent=discogs_node,
        critical=False
    )

    # Check price is under $50
    price_value = extract_price_value(vinyl_info.price_text)
    price_ok = price_value is not None and price_value < 50
    evaluator.add_custom_node(
        result=bool(price_ok),
        id="discogs_perception_price",
        desc="Verify the item is priced under $50",
        parent=discogs_node,
        critical=False
    )

    # Check purchase link is provided
    link_ok = bool(vinyl_info.purchase_link and vinyl_info.purchase_link.strip())
    link_is_discogs = is_discogs_link(vinyl_info.purchase_link)
    evaluator.add_custom_node(
        result=bool(link_ok and link_is_discogs),
        id="discogs_perception_purchase_link",
        desc="Provide a valid Discogs purchase link",
        parent=discogs_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
