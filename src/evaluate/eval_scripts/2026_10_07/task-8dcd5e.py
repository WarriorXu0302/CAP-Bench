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
TASK_ID = "task-8dcd5e"
TASK_DESCRIPTION = 'I’m an event promoter looking for promising indie bands. First, go to **Eventbrite**, search for **Music** events in **Los Angeles**, and use the date filter to show only events happening **this weekend**. Prioritize finding three indie band headline shows that are **not Sold Out** (avoid multi-artist festivals or DJ parties). If fewer than three are available, record the actual number found and continue. Note the name of your top-priority band choice.\n\nTake that band name to **Spotify** to verify popularity. Open the artist profile page, switch to the **Albums** or **Discography** tab, and check whether they have released any album with “Live” in the title (output **Yes/No**). Then check the **Monthly Listeners** on the main page, which must be above **10,000**. If it is below 10,000, return to Eventbrite and choose the next band until one meets the requirement. If all available candidates in this round are below 10,000, state that in the result and stop. Once a band qualifies, record the #1 song in their **Popular** list.\n\nFinally, go to **YouTube** to validate live performance strength. Search for a live version of that top song (keyword: **“Song Name Live”**), and use filters to show only videos with duration **Over 4 minutes**. Return up to 5 candidate results (or all if fewer), sort by view count from highest to lowest, and select the top one as the final result. Also include the titles and view counts of the #2 and #3 videos for comparison (if fewer than 3 exist, provide the actual number available).\n\n**Output:** final selected band name, Eventbrite event link, Spotify monthly listeners, whether they have a Live album, Spotify Top 1 song name, YouTube video title, duration, view count, and link; plus the YouTube #2/#3 titles and view counts (if available).'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventbriteInfo(BaseModel):
    """Eventbrite event details extracted from the answer"""
    band_name: Optional[str] = None
    event_link: Optional[str] = None
    event_location: Optional[str] = None
    event_date: Optional[str] = None
    sold_out_status: Optional[str] = None


class SpotifyInfo(BaseModel):
    """Spotify artist details extracted from the answer"""
    monthly_listeners: Optional[str] = None
    has_live_album: Optional[str] = None
    top_song_name: Optional[str] = None


class YouTubeInfo(BaseModel):
    """YouTube video details extracted from the answer"""
    video_title: Optional[str] = None
    video_duration: Optional[str] = None
    video_view_count: Optional[str] = None
    video_link: Optional[str] = None
    second_video_title: Optional[str] = None
    second_video_views: Optional[str] = None
    third_video_title: Optional[str] = None
    third_video_views: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eventbrite_from_answer() -> str:
    return """
Extract the Eventbrite event details from the answer:

Return:
- band_name: the name of the selected indie band
- event_link: the Eventbrite event URL
- event_location: the location mentioned (should be Los Angeles or LA)
- event_date: the date or time frame mentioned (should be this weekend)
- sold_out_status: any mention of whether the event is sold out or not

If any field is missing in the answer, set it to null.
"""


def prompt_extract_spotify_from_answer() -> str:
    return """
Extract the Spotify artist details from the answer:

Return:
- monthly_listeners: the number of monthly listeners as stated (include any formatting like commas)
- has_live_album: whether the artist has a Live album (should be "Yes" or "No")
- top_song_name: the #1 song from the Popular list

If any field is missing, set it to null.
"""


def prompt_extract_youtube_from_answer() -> str:
    return """
Extract the YouTube video details from the answer:

Return:
- video_title: the title of the top selected video
- video_duration: the duration of the top video
- video_view_count: the view count of the top video
- video_link: the YouTube URL of the top video
- second_video_title: the title of the #2 video (if available)
- second_video_views: the view count of the #2 video (if available)
- third_video_title: the title of the #3 video (if available)
- third_video_views: the view count of the #3 video (if available)

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
    # Remove commas and extract number
    cleaned = re.sub(r'[,\s]', '', text)
    m = re.findall(r'(\d+(?:\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://', text))


def looks_like_duration_over_4min(text: Optional[str]) -> bool:
    if not text:
        return False
    # Match patterns like "5:30", "4:15", "12:00", etc.
    m = re.search(r'(\d+):(\d+)', text)
    if not m:
        return False
    minutes = int(m.group(1))
    return minutes >= 4


def mentions_los_angeles(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['los angeles', 'la', 'l.a.'])


def mentions_this_weekend(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['this weekend', 'weekend', 'saturday', 'sunday'])


def mentions_music(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['music'])


def mentions_not_sold_out(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for explicit mentions that it's NOT sold out
    if has_any_ci(text, ['not sold out', 'available', 'tickets available']):
        return True
    # Absence of "sold out" when checking status is also acceptable
    return not has_any_ci(text, ['sold out'])


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
    eventbrite_info = await evaluator.extract(
        prompt=prompt_extract_eventbrite_from_answer(),
        template_class=EventbriteInfo,
        extraction_name="eventbrite_info"
    )

    spotify_info = await evaluator.extract(
        prompt=prompt_extract_spotify_from_answer(),
        template_class=SpotifyInfo,
        extraction_name="spotify_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_from_answer(),
        template_class=YouTubeInfo,
        extraction_name="youtube_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite music events search in Los Angeles for this weekend",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location filtering (Los Angeles)
    location_ok = mentions_los_angeles(answer) or mentions_los_angeles(eventbrite_info.event_location)
    evaluator.add_custom_node(
        result=bool(location_ok),
        id="eventbrite_location_filter",
        desc="[Action Node] eventbrite.com:F1:A12 - Filter events by Los Angeles location",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date filtering (this weekend)
    date_ok = mentions_this_weekend(answer) or mentions_this_weekend(eventbrite_info.event_date)
    evaluator.add_custom_node(
        result=bool(date_ok),
        id="eventbrite_date_filter",
        desc="[Action Node] eventbrite.com:F1:A1 - Apply date filter to show only this weekend events",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A13 - Category navigation (Music)
    music_ok = mentions_music(answer)
    evaluator.add_custom_node(
        result=bool(music_ok),
        id="eventbrite_music_category",
        desc="[Action Node] eventbrite.com:F1:A13 - Navigate to or filter for Music category events",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Status awareness (not Sold Out)
    sold_out_check_ok = mentions_not_sold_out(answer)
    evaluator.add_custom_node(
        result=bool(sold_out_check_ok),
        id="eventbrite_not_sold_out",
        desc="[Perception Node] eventbrite.com:F1:P1 - Verify selected event is not Sold Out",
        parent=eventbrite_node,
        critical=False
    )

    # Band name and event link extracted
    band_name_ok = bool(eventbrite_info.band_name and eventbrite_info.band_name.strip())
    event_link_ok = looks_like_url(eventbrite_info.event_link)
    evaluator.add_custom_node(
        result=bool(band_name_ok and event_link_ok),
        id="eventbrite_band_and_link",
        desc="Band name and Eventbrite event link provided",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 Spotify section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify artist verification and popularity check",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F2:A10 - Click into artist detail page
    spotify_access_ok = has_any_ci(answer, ['spotify'])
    evaluator.add_custom_node(
        result=bool(spotify_access_ok),
        id="spotify_artist_detail",
        desc="[Action Node] open.spotify.com:F2:A10 - Navigate to the artist's detail page on Spotify",
        parent=spotify_node,
        critical=False
    )

    # [Action Node] open.spotify.com:F2:A3 - Tab switch (Albums/Discography)
    has_live_album_value = spotify_info.has_live_album
    tab_switch_ok = bool(has_live_album_value and has_live_album_value.strip().lower() in ['yes', 'no'])
    evaluator.add_custom_node(
        result=bool(tab_switch_ok),
        id="spotify_albums_tab",
        desc="[Action Node] open.spotify.com:F2:A3 - Switch to Albums or Discography tab to check for Live album",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F2:P23 - Data awareness (Monthly Listeners > 10,000)
    listeners_num = extract_number(spotify_info.monthly_listeners)
    listeners_ok = listeners_num is not None and listeners_num > 10000
    evaluator.add_custom_node(
        result=bool(listeners_ok),
        id="spotify_monthly_listeners",
        desc="[Perception Node] open.spotify.com:F2:P23 - Verify Monthly Listeners count exceeds 10,000",
        parent=spotify_node,
        critical=False
    )

    # Top song extracted
    top_song_ok = bool(spotify_info.top_song_name and spotify_info.top_song_name.strip())
    evaluator.add_custom_node(
        result=bool(top_song_ok),
        id="spotify_top_song",
        desc="Top song from Popular list identified",
        parent=spotify_node,
        critical=False
    )

    # 3.3 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube live performance video search and validation",
        parent=root,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Content matching (song name + "Live")
    video_title_ok = bool(youtube_info.video_title and youtube_info.video_title.strip())
    live_keyword_ok = has_any_ci(youtube_info.video_title, ['live']) or has_any_ci(answer, ['live'])
    content_match_ok = video_title_ok and live_keyword_ok
    evaluator.add_custom_node(
        result=bool(content_match_ok),
        id="youtube_live_search",
        desc="[Perception Node] youtube.com:F1:P4 - Search results match song name with 'Live' keyword",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Advanced filtering (duration Over 4 minutes)
    duration_ok = looks_like_duration_over_4min(youtube_info.video_duration)
    evaluator.add_custom_node(
        result=bool(duration_ok),
        id="youtube_duration_filter",
        desc="[Action Node] youtube.com:F9:A4 - Apply duration filter to show only videos over 4 minutes",
        parent=youtube_node,
        critical=False
    )

    # Top video details complete
    views_ok = contains_digits(youtube_info.video_view_count)
    link_ok = looks_like_url(youtube_info.video_link)
    top_video_ok = video_title_ok and duration_ok and views_ok and link_ok
    evaluator.add_custom_node(
        result=bool(top_video_ok),
        id="youtube_top_video_complete",
        desc="Top YouTube video with title, duration, view count, and link provided",
        parent=youtube_node,
        critical=False
    )

    # Comparison videos (2nd and 3rd)
    has_second = bool(youtube_info.second_video_title and youtube_info.second_video_title.strip())
    has_third = bool(youtube_info.third_video_title and youtube_info.third_video_title.strip())
    comparison_ok = has_second or has_third
    evaluator.add_custom_node(
        result=bool(comparison_ok),
        id="youtube_comparison_videos",
        desc="Additional videos (#2 and/or #3) provided for comparison",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
