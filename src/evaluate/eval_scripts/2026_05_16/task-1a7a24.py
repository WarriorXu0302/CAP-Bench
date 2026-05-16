import asyncio
import logging
from typing import Optional, List, Dict, Any
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-1a7a24"
TASK_DESCRIPTION = """
I am producing a "2024 Indie Game Review" video and need to gather material for three games: Pacific Drive, Balatro, and Animal Well. For each game, please perform the following steps in order:

1.  First, on YouTube, search for the game's "Official Launch Trailer". Filter for the video with the highest views that is published by the official developer or publisher channel, and ensure it supports 4K or 1080p resolution.
2.  Next, go to Metacritic and search for the game's PC version page. Record its User Score. Then, find a user review with a score of 10 and extract a core statement from it.
3.  Finally, on Twitch, search for the game's category. Navigate to the "Clips" section, set the filter to "All Time", and identify the clip with the highest view count.

Please output the following for each of the three games: YouTube Trailer Title, Publishing Channel, Metacritic User Score, Selected Review Summary, Twitch Most Viewed Clip Title, View Count, Streamer Name, and all relevant detail page links.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted answer content                                    #
# --------------------------------------------------------------------------- #
class YouTubeInfo(BaseModel):
    title: Optional[str] = None
    channel: Optional[str] = None
    url: Optional[str] = None
    resolution: Optional[str] = None  # Should include "4K" or "1080p"


class MetacriticInfo(BaseModel):
    pc_url: Optional[str] = None
    user_score: Optional[str] = None  # Keep raw; will parse to float
    review_summary: Optional[str] = None  # Core statement from a 10/10 review
    review_url: Optional[str] = None


class TwitchInfo(BaseModel):
    category_url: Optional[str] = None
    clip_title: Optional[str] = None
    clip_views: Optional[str] = None  # Raw view count string
    clip_streamer: Optional[str] = None
    clip_url: Optional[str] = None
    clips_filter: Optional[str] = None  # Expect "All Time", if provided


class GameEntry(BaseModel):
    game: Optional[str] = None
    youtube: Optional[YouTubeInfo] = None
    metacritic: Optional[MetacriticInfo] = None
    twitch: Optional[TwitchInfo] = None


class GamesPayload(BaseModel):
    games: List[GameEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_games_payload() -> str:
    return """
Extract the requested information for the three games "Pacific Drive", "Balatro", and "Animal Well" from the answer.

For each of these games, if present, return the following fields:
- game: the game name
- youtube:
  - title: YouTube trailer title
  - channel: publishing channel name
  - url: the direct YouTube video URL
  - resolution: the video resolution indicator mentioned (e.g., "4K", "2160p", "1080p", etc.)
- metacritic:
  - pc_url: the direct URL of the game's PC version page on Metacritic
  - user_score: the user score number (as shown on the page)
  - review_summary: a short summary statement extracted from a user review rated 10/10
  - review_url: URL of the specific review page or user reviews section
- twitch:
  - category_url: the Twitch category page URL (if mentioned)
  - clip_title: the most viewed clip title
  - clip_views: the view count string (e.g., "1,234,567" or "2.3M")
  - clip_streamer: the streamer name for the clip
  - clip_url: the direct URL of the clip detail page
  - clips_filter: the selected time range filter for clips (e.g., "All Time") if mentioned

Return all of this data in the specified JSON format.
"""


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def url_domain_contains(url: Optional[str], substrings: List[str]) -> bool:
    if not url or not is_valid_url(url):
        return False
    try:
        host = urlparse(url).netloc.lower()
        return any(x in host for x in substrings)
    except Exception:
        return False


def is_youtube_url(url: Optional[str]) -> bool:
    return url_domain_contains(url, ["youtube.com", "youtu.be"])


def is_twitch_clip_url(url: Optional[str]) -> bool:
    if not is_valid_url(url):
        return False
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    # Accept both clips.twitch.tv/<slug> and twitch.tv/<channel>/clip/<slug>
    if "clips.twitch.tv" in host:
        return True
    if "twitch.tv" in host and "/clip/" in path:
        return True
    return False


def is_advertisement_url(url: Optional[str]) -> bool:
    # Heuristic to exclude ad/redirect links
    if not is_valid_url(url):
        return False
    host = urlparse(url).netloc.lower()
    return any(x in host for x in ["googleadservices", "doubleclick", "adservice", "adserver", "ad." ])


def is_official_channel(channel: Optional[str], game_name: Optional[str]) -> bool:
    ch = normalize_text(channel)
    if not ch:
        return False
    # Lenient heuristic: contains 'official' or ends with 'games'/'studios' or well-known publisher markers
    if "official" in ch:
        return True
    if any(tok in ch for tok in ["games", "studios", "publisher", "interactive", "entertainment", "devolver", "nintendo", "playstation", "xbox", "sony", "valve", "focus", "raw fury", "annapurna", "humble", "kwang", "bandai", "505", "tinybuild"]):
        return True
    # If channel name includes the game name exactly (lenient)
    if game_name and normalize_text(game_name) in ch:
        return True
    return False


def has_required_resolution(resolution: Optional[str]) -> bool:
    r = normalize_text(resolution)
    return ("4k" in r) or ("2160" in r) or ("1080" in r)


def is_metacritic_pc_url(url: Optional[str]) -> bool:
    if not url or not is_valid_url(url):
        return False
    if "metacritic.com" not in url.lower():
        return False
    lower = url.lower()
    # Accept URLs that contain /game/pc or /pc/
    if "/game/pc" in lower or "/pc/" in lower or "/pc" in lower:
        return True
    # Accept query param pattern for platform
    if "platform=pc" in lower:
        return True
    return False


def parse_user_score(user_score: Optional[str]) -> Optional[float]:
    if not user_score:
        return None
    s = normalize_text(user_score)
    # Handle "tbd" or non-numeric gracefully
    if "tbd" in s or "n/a" in s:
        return None
    # Extract numeric float
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    try:
        val = float(m.group(1))
        return val
    except Exception:
        return None


def parse_view_count(views: Optional[str]) -> Optional[float]:
    if views is None:
        return None
    s = normalize_text(views).replace(",", "").strip()
    # Handle suffix K/M
    try:
        if s.endswith("k"):
            return float(s[:-1]) * 1_000
        if s.endswith("m"):
            return float(s[:-1]) * 1_000_000
        # Remove non-digits except dot
        m = re.search(r"(\d+(\.\d+)?)", s)
        if not m:
            return None
        return float(m.group(1))
    except Exception:
        # Fallback: try to remove all non-digit
        digits = re.sub(r"[^\d.]", "", s)
        try:
            return float(digits) if digits else None
        except Exception:
            return None


def non_empty(s: Optional[str]) -> bool:
    return bool((s or "").strip())


def find_game_entry(games: List[GameEntry], target_name: str) -> Optional[GameEntry]:
    tn = normalize_text(target_name)
    for g in games:
        gn = normalize_text(g.game)
        if gn == tn or tn in gn or gn in tn:
            return g
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
    """
    # 1. Set up evaluator
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

    # 2. Extract structured data for the three games from the answer
    games_payload = await evaluator.extract(
        prompt=prompt_extract_games_payload(),
        template_class=GamesPayload,
        extraction_name="games_payload"
    )

    all_games = games_payload.games if games_payload and games_payload.games else []

    # Prepare entries for each expected game
    expected_games = ["Pacific Drive", "Balatro", "Animal Well"]
    selected_entries: List[GameEntry] = []
    for gname in expected_games:
        entry = find_game_entry(all_games, gname)
        if not entry:
            # Create a placeholder to allow partial credit on available parts
            entry = GameEntry(game=gname, youtube=YouTubeInfo(), metacritic=MetacriticInfo(), twitch=TwitchInfo())
        selected_entries.append(entry)

    # 3. Build verification tree
    # Top-level parallel nodes for each game
    for entry in selected_entries:
        game_name = entry.game or "Unknown Game"
        game_node = evaluator.add_sequential(
            id=f"game_{normalize_text(game_name).replace(' ', '_')}",
            desc=f"Verify collected materials for '{game_name}' across YouTube, Metacritic (PC), and Twitch",
            parent=root,
            critical=False
        )

        # --- YouTube checks ---
        yt = entry.youtube or YouTubeInfo()
        yt_node = evaluator.add_parallel(
            id=f"{normalize_text(game_name)}_youtube",
            desc=f"YouTube verification for '{game_name}'",
            parent=game_node,
            critical=False
        )

        # [Perception Node] youtube.com:F1:P4 - Official channel check
        yt_official_node = evaluator.add_custom_node(
            result=is_official_channel(yt.channel, game_name),
            id=f"{normalize_text(game_name)}_yt_official_channel",
            desc="[Perception Node] youtube.com:F1:P4 - Check publishing channel indicates official developer/publisher (e.g., contains 'official' or org keywords)",
            parent=yt_node,
            critical=False
        )

        # [Perception Node] youtube.com:F1:P5 - Exclude ads/promoted results
        yt_not_ad_node = evaluator.add_custom_node(
            result=is_youtube_url(yt.url) and not is_advertisement_url(yt.url),
            id=f"{normalize_text(game_name)}_yt_not_ad",
            desc="[Perception Node] youtube.com:F1:P5 - Ensure selected YouTube URL is not an advertisement/redirect link",
            parent=yt_node,
            critical=False
        )

        # [Action Node] youtube.com:F1:A22 - Click into details (URL validity)
        yt_detail_url_node = evaluator.add_custom_node(
            result=is_youtube_url(yt.url),
            id=f"{normalize_text(game_name)}_yt_detail_url",
            desc="[Action Node] youtube.com:F1:A22 - Validate direct YouTube video page URL is provided",
            parent=yt_node,
            critical=False
        )

        # General: Resolution check (4K or 1080p)
        yt_resolution_node = evaluator.add_custom_node(
            result=has_required_resolution(yt.resolution),
            id=f"{normalize_text(game_name)}_yt_resolution",
            desc="Check YouTube trailer supports 4K or 1080p resolution (lenient match)",
            parent=yt_node,
            critical=False
        )

        # --- Metacritic checks ---
        mc = entry.metacritic or MetacriticInfo()
        mc_node = evaluator.add_parallel(
            id=f"{normalize_text(game_name)}_metacritic",
            desc=f"Metacritic (PC) verification for '{game_name}'",
            parent=game_node,
            critical=False
        )

        # [Action Node] metacritic.comgame:F1:A1 - Ensure PC platform page
        mc_pc_node = evaluator.add_custom_node(
            result=is_metacritic_pc_url(mc.pc_url),
            id=f"{normalize_text(game_name)}_mc_pc_url",
            desc="[Action Node] metacritic.comgame:F1:A1 - Verify Metacritic URL points to PC platform page",
            parent=mc_node,
            critical=False
        )

        # [Perception Node] metacritic.comgame:F1:P1 - User Score numeric 0-10
        user_score_val = parse_user_score(mc.user_score)
        mc_user_score_node = evaluator.add_custom_node(
            result=(user_score_val is not None and 0.0 <= user_score_val <= 10.0),
            id=f"{normalize_text(game_name)}_mc_user_score",
            desc="[Perception Node] metacritic.comgame:F1:P1 - Validate User Score is a number between 0 and 10",
            parent=mc_node,
            critical=False
        )

        # [Action Node] metacritic.comgame:F1:A6 - Click into user review details
        has_review_summary = non_empty(mc.review_summary)
        mc_review_detail_node = evaluator.add_custom_node(
            result=has_review_summary and (is_valid_url(mc.review_url) and "metacritic.com" in (mc.review_url or "").lower()),
            id=f"{normalize_text(game_name)}_mc_review_detail",
            desc="[Action Node] metacritic.comgame:F1:A6 - Ensure user review (score 10) summary is extracted and a valid review-related URL is provided",
            parent=mc_node,
            critical=False
        )

        # --- Twitch checks ---
        tw = entry.twitch or TwitchInfo()
        tw_node = evaluator.add_parallel(
            id=f"{normalize_text(game_name)}_twitch",
            desc=f"Twitch Clips verification for '{game_name}'",
            parent=game_node,
            critical=False
        )

        # [Action Node] twitch.tv:F3:A11 - Highest viewed clip identification (lenient threshold)
        views_val = parse_view_count(tw.clip_views)
        tw_highest_node = evaluator.add_custom_node(
            result=(views_val is not None and views_val >= 5000),  # lenient threshold for "significantly high"
            id=f"{normalize_text(game_name)}_tw_highest_views",
            desc="[Action Node] twitch.tv:F3:A11 - Check selected clip has high view count (leniently > 5,000)",
            parent=tw_node,
            critical=False
        )

        # [Action Node] twitch.tv:F3:A2 - All Time filter applied
        answer_has_all_time = "all time" in normalize_text(answer)
        tw_alltime_node = evaluator.add_custom_node(
            result=(normalize_text(tw.clips_filter) == "all time") or answer_has_all_time,
            id=f"{normalize_text(game_name)}_tw_all_time_filter",
            desc="[Action Node] twitch.tv:F3:A2 - Verify 'All Time' clips filter is indicated (in field or answer text)",
            parent=tw_node,
            critical=False
        )

        # [Action Node] twitch.tv:F7:A2 - Clip detail page URL validity
        tw_clip_url_node = evaluator.add_custom_node(
            result=is_twitch_clip_url(tw.clip_url),
            id=f"{normalize_text(game_name)}_tw_clip_url",
            desc="[Action Node] twitch.tv:F7:A2 - Validate the URL points to a specific Twitch clip page",
            parent=tw_node,
            critical=False
        )

        # General: Presence of clip title and streamer
        tw_title_streamer_node = evaluator.add_custom_node(
            result=non_empty(tw.clip_title) and non_empty(tw.clip_streamer),
            id=f"{normalize_text(game_name)}_tw_title_streamer",
            desc="Check Twitch clip title and streamer name are provided",
            parent=tw_node,
            critical=False
        )

    # 4. Return structured result
    return evaluator.get_summary()