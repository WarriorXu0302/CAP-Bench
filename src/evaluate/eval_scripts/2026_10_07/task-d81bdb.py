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
TASK_ID = "task-d81bdb"
TASK_DESCRIPTION = "I am researching the 2025 NBA player trade market and would like to conduct an in-depth analysis of Anthony Davis's trade value for the Los Angeles Lakers.\n\nFirst, go to Basketball Reference to look up his 2024-25 season's per-game statistics (points, rebounds, assists, PER) and career advanced statistics. Pay particular attention to the PER, Win Shares, and VORP metrics within the Advanced Stats section.\n\nNext, go to the official NBA website to confirm his contract status, including current salary, contract expiration year, and any team or player options.\n\nThen, search ESPN for 'Anthony Davis trade rumors 2025'. Find news articles published within the last month to check for any trade rumors, injury reports, or team dynamics. Note down the titles and key points of 2-3 significant articles.\n\nFinally, go to YouTube and search for 'Anthony Davis trade analysis 2025'. Sort by view count and find the top 2 expert analysis videos (each lasting over 15 minutes). Record the video titles, channel names, view counts, and video URLs.\n\nOutput:\nAD's per-game stats for the 2024-25 season (points, rebounds, assists, PER), career PER, Win Shares, VORP, and the Basketball Reference data page URL;\nContract length, current annual salary, contract expiration year, presence of any options, and the NBA official player page URL;\nESPN news article titles, publication dates, core takeaways (trade possibility/injury status), and news article URLs;\nYouTube video titles, channel names, view counts, durations, and video URLs."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BasketballRefStats(BaseModel):
    """Basketball Reference stats extracted from the answer"""
    points_per_game: Optional[str] = None
    rebounds_per_game: Optional[str] = None
    assists_per_game: Optional[str] = None
    per_season: Optional[str] = None
    per_career: Optional[str] = None
    win_shares: Optional[str] = None
    vorp: Optional[str] = None
    bbref_url: Optional[str] = None


class NBAContractInfo(BaseModel):
    """NBA.com contract information extracted from the answer"""
    contract_length: Optional[str] = None
    current_salary: Optional[str] = None
    expiration_year: Optional[str] = None
    options: Optional[str] = None
    nba_url: Optional[str] = None


class ESPNArticle(BaseModel):
    """Single ESPN article information"""
    title: Optional[str] = None
    publication_date: Optional[str] = None
    key_points: Optional[str] = None
    url: Optional[str] = None


class ESPNArticles(BaseModel):
    """ESPN articles extracted from the answer"""
    articles: List[ESPNArticle] = Field(default_factory=list)


class YouTubeVideo(BaseModel):
    """Single YouTube video information"""
    title: Optional[str] = None
    channel_name: Optional[str] = None
    view_count: Optional[str] = None
    duration: Optional[str] = None
    url: Optional[str] = None


class YouTubeVideos(BaseModel):
    """YouTube videos extracted from the answer"""
    videos: List[YouTubeVideo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bbref_stats() -> str:
    return """
Extract the Basketball Reference statistics for Anthony Davis from the answer.

Return:
- points_per_game: 2024-25 season per-game points exactly as stated
- rebounds_per_game: 2024-25 season per-game rebounds exactly as stated
- assists_per_game: 2024-25 season per-game assists exactly as stated
- per_season: 2024-25 season PER value exactly as stated
- per_career: Career PER value exactly as stated
- win_shares: Win Shares value exactly as stated
- vorp: VORP value exactly as stated
- bbref_url: Basketball Reference URL for Anthony Davis page

If any field is missing, set it to null.
"""


def prompt_extract_nba_contract() -> str:
    return """
Extract the NBA.com contract information for Anthony Davis from the answer.

Return:
- contract_length: Contract length/years exactly as stated
- current_salary: Current annual salary exactly as stated
- expiration_year: Contract expiration year exactly as stated
- options: Any team or player options exactly as stated
- nba_url: NBA.com player page URL

If any field is missing, set it to null.
"""


def prompt_extract_espn_articles() -> str:
    return """
Extract the ESPN news articles about Anthony Davis from the answer.

Return a list of articles, each containing:
- title: Article title exactly as stated
- publication_date: Publication date exactly as stated
- key_points: Core takeaways about trade possibility, injury status, etc.
- url: Article URL

Return an empty list if no articles are found.
"""


def prompt_extract_youtube_videos() -> str:
    return """
Extract the YouTube videos about Anthony Davis trade analysis from the answer.

Return a list of videos, each containing:
- title: Video title exactly as stated
- channel_name: Channel name exactly as stated
- view_count: View count exactly as stated
- duration: Video duration exactly as stated
- url: Video URL

Return an empty list if no videos are found.
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


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return ci_contains(text, domain) and (ci_contains(text, 'http') or ci_contains(text, 'www'))


def mentions_season_2024_25(answer: str) -> bool:
    patterns = [r'2024-25', r'2024/25', r'24-25', r'2024-2025']
    return any(re.search(p, answer) for p in patterns)


def mentions_advanced_stats(answer: str) -> bool:
    return has_any_ci(answer, ['advanced stats', 'advanced statistics', 'per', 'win shares', 'vorp'])


def mentions_contract_details(answer: str) -> bool:
    return has_any_ci(answer, ['salary', 'contract', 'expiration', 'option'])


def mentions_espn_search(answer: str) -> bool:
    return has_any_ci(answer, ['espn']) and has_any_ci(answer, ['trade', 'rumors'])


def mentions_youtube_search(answer: str) -> bool:
    return has_any_ci(answer, ['youtube']) and has_any_ci(answer, ['trade analysis', 'analysis'])


def count_articles(articles: List[ESPNArticle]) -> int:
    return sum(1 for a in articles if a.title or a.url)


def count_videos(videos: List[YouTubeVideo]) -> int:
    return sum(1 for v in videos if v.title or v.url)


def check_duration_over_15min(duration: Optional[str]) -> bool:
    if not duration:
        return False
    # Try to extract minutes from duration string
    m = re.findall(r'(\d+)\s*(?:min|minute|m)', duration.lower())
    if m:
        try:
            minutes = int(m[0])
            return minutes >= 15
        except Exception:
            pass
    return False


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
    bbref_stats = await evaluator.extract(
        prompt=prompt_extract_bbref_stats(),
        template_class=BasketballRefStats,
        extraction_name="basketball_reference_stats"
    )

    nba_contract = await evaluator.extract(
        prompt=prompt_extract_nba_contract(),
        template_class=NBAContractInfo,
        extraction_name="nba_contract_info"
    )

    espn_articles = await evaluator.extract(
        prompt=prompt_extract_espn_articles(),
        template_class=ESPNArticles,
        extraction_name="espn_articles"
    )

    youtube_videos = await evaluator.extract(
        prompt=prompt_extract_youtube_videos(),
        template_class=YouTubeVideos,
        extraction_name="youtube_videos"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Basketball Reference section
    bbref_node = evaluator.add_sequential(
        id="basketball_reference_section",
        desc="Basketball Reference statistics for Anthony Davis 2024-25 season and career",
        parent=root,
        critical=False
    )

    # [Action Node] basketball-reference.com:F1:A13 - Search for player
    bbref_search_ok = has_any_ci(answer, ['basketball reference', 'bbref']) and has_any_ci(answer, ['anthony davis', 'AD'])
    evaluator.add_custom_node(
        result=bool(bbref_search_ok),
        id="bbref_action_search",
        desc="[Action Node] basketball-reference.com:F1:A13 - Search for Anthony Davis on Basketball Reference",
        parent=bbref_node,
        critical=False
    )

    # [Action Node] basketball-reference.com:F1:A4 - Select 2024-25 season
    season_data_ok = mentions_season_2024_25(answer) and contains_digits(bbref_stats.points_per_game or "")
    evaluator.add_custom_node(
        result=bool(season_data_ok),
        id="bbref_action_season_select",
        desc="[Action Node] basketball-reference.com:F1:A4 - Select 2024-25 season from dropdown",
        parent=bbref_node,
        critical=False
    )

    # [Action Node] basketball-reference.com:F1:A2 - Switch to Advanced Stats tab
    advanced_stats_ok = mentions_advanced_stats(answer)
    evaluator.add_custom_node(
        result=bool(advanced_stats_ok),
        id="bbref_action_advanced_tab",
        desc="[Action Node] basketball-reference.com:F1:A2 - Switch to Advanced Stats tab",
        parent=bbref_node,
        critical=False
    )

    # [Perception Node] basketball-reference.com:F1:P1 - Extract per-game stats
    pergame_complete = all([
        contains_digits(bbref_stats.points_per_game or ""),
        contains_digits(bbref_stats.rebounds_per_game or ""),
        contains_digits(bbref_stats.assists_per_game or ""),
        contains_digits(bbref_stats.per_season or "")
    ])
    evaluator.add_custom_node(
        result=bool(pergame_complete),
        id="bbref_perception_pergame",
        desc="[Perception Node] basketball-reference.com:F1:P1 - Extract per-game stats (points, rebounds, assists, PER)",
        parent=bbref_node,
        critical=False
    )

    # [Perception Node] basketball-reference.com:F1:P8 - Extract advanced stats
    advanced_complete = all([
        contains_digits(bbref_stats.per_career or ""),
        contains_digits(bbref_stats.win_shares or ""),
        contains_digits(bbref_stats.vorp or "")
    ])
    evaluator.add_custom_node(
        result=bool(advanced_complete),
        id="bbref_perception_advanced",
        desc="[Perception Node] basketball-reference.com:F1:P8 - Extract career PER, Win Shares, and VORP",
        parent=bbref_node,
        critical=False
    )

    # URL validation
    bbref_url_ok = looks_like_url(bbref_stats.bbref_url, 'basketball-reference.com')
    evaluator.add_custom_node(
        result=bool(bbref_url_ok),
        id="bbref_url_present",
        desc="Basketball Reference URL is provided",
        parent=bbref_node,
        critical=False
    )

    # 3.2 NBA.com section
    nba_node = evaluator.add_sequential(
        id="nba_com_section",
        desc="NBA.com contract information for Anthony Davis",
        parent=root,
        critical=False
    )

    # [Action Node] nba.com:F5:A20 - Click player card to enter profile
    nba_action_ok = has_any_ci(answer, ['nba.com', 'nba official']) and has_any_ci(answer, ['anthony davis', 'AD'])
    evaluator.add_custom_node(
        result=bool(nba_action_ok),
        id="nba_action_player_card",
        desc="[Action Node] nba.com:F5:A20 - Navigate to Anthony Davis player profile on NBA.com",
        parent=nba_node,
        critical=False
    )

    # [Action Node] nba.com:F5:A3 - Switch to contract tab
    contract_context_ok = mentions_contract_details(answer)
    evaluator.add_custom_node(
        result=bool(contract_context_ok),
        id="nba_action_contract_tab",
        desc="[Action Node] nba.com:F5:A3 - Switch to contract information tab",
        parent=nba_node,
        critical=False
    )

    # [Perception Node] nba.com:F5:P4 - Extract contract details
    contract_complete = all([
        nba_contract.contract_length is not None,
        nba_contract.current_salary is not None and contains_digits(nba_contract.current_salary),
        nba_contract.expiration_year is not None and contains_digits(nba_contract.expiration_year)
    ])
    evaluator.add_custom_node(
        result=bool(contract_complete),
        id="nba_perception_contract",
        desc="[Perception Node] nba.com:F5:P4 - Extract contract length, salary, expiration year, and options",
        parent=nba_node,
        critical=False
    )

    # URL validation
    nba_url_ok = looks_like_url(nba_contract.nba_url, 'nba.com')
    evaluator.add_custom_node(
        result=bool(nba_url_ok),
        id="nba_url_present",
        desc="NBA.com player page URL is provided",
        parent=nba_node,
        critical=False
    )

    # 3.3 ESPN section
    espn_node = evaluator.add_sequential(
        id="espn_section",
        desc="ESPN news articles about Anthony Davis trade rumors",
        parent=root,
        critical=False
    )

    # [Action Node] espn.com:F1:A13 - Navigate to NBA section
    espn_search_ok = mentions_espn_search(answer)
    evaluator.add_custom_node(
        result=bool(espn_search_ok),
        id="espn_action_navigate",
        desc="[Action Node] espn.com:F1:A13 - Navigate to ESPN NBA section and search for Anthony Davis trade rumors",
        parent=espn_node,
        critical=False
    )

    # [Action Node] espn.com:F1:A12 - Click news cards
    article_count = count_articles(espn_articles.articles)
    articles_found = article_count >= 2
    evaluator.add_custom_node(
        result=bool(articles_found),
        id="espn_action_click_cards",
        desc="[Action Node] espn.com:F1:A12 - Click on news article cards to read details",
        parent=espn_node,
        critical=False
    )

    # [Perception Node] espn.com:F1:P8 - Identify publication dates
    dates_present = all(a.publication_date for a in espn_articles.articles if a.title)
    evaluator.add_custom_node(
        result=bool(dates_present and article_count >= 2),
        id="espn_perception_dates",
        desc="[Perception Node] espn.com:F1:P8 - Identify article publication dates within last month",
        parent=espn_node,
        critical=False
    )

    # [Perception Node] espn.com:F1:P9 - Identify content tags
    key_points_present = all(a.key_points for a in espn_articles.articles if a.title)
    evaluator.add_custom_node(
        result=bool(key_points_present and article_count >= 2),
        id="espn_perception_content",
        desc="[Perception Node] espn.com:F1:P9 - Extract key points about trade/injury/team dynamics",
        parent=espn_node,
        critical=False
    )

    # URLs validation
    article_urls_ok = all(looks_like_url(a.url, 'espn.com') for a in espn_articles.articles if a.title)
    evaluator.add_custom_node(
        result=bool(article_urls_ok and article_count >= 2),
        id="espn_urls_present",
        desc="ESPN article URLs are provided for 2-3 articles",
        parent=espn_node,
        critical=False
    )

    # 3.4 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube videos about Anthony Davis trade analysis 2025",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click video cards
    video_count = count_videos(youtube_videos.videos)
    videos_found = video_count >= 2
    evaluator.add_custom_node(
        result=bool(videos_found),
        id="youtube_action_click_video",
        desc="[Action Node] youtube.com:F1:A22 - Click on video cards to view details",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll to load more results
    youtube_search_ok = mentions_youtube_search(answer)
    evaluator.add_custom_node(
        result=bool(youtube_search_ok),
        id="youtube_action_scroll",
        desc="[Action Node] youtube.com:F1:A69 - Scroll through search results to find top videos by view count",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Identify video thumbnail relevance
    titles_relevant = all(
        has_any_ci(v.title, ['anthony davis', 'AD']) and has_any_ci(v.title, ['trade', 'analysis'])
        for v in youtube_videos.videos if v.title
    )
    evaluator.add_custom_node(
        result=bool(titles_relevant and video_count >= 2),
        id="youtube_perception_thumbnail",
        desc="[Perception Node] youtube.com:F1:P4 - Identify video content relevance from thumbnails and titles",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P30 - Compare and filter videos
    view_counts_present = all(contains_digits(v.view_count or "") for v in youtube_videos.videos if v.title)
    durations_valid = all(check_duration_over_15min(v.duration) for v in youtube_videos.videos if v.title)
    evaluator.add_custom_node(
        result=bool(view_counts_present and durations_valid and video_count >= 2),
        id="youtube_perception_compare",
        desc="[Perception Node] youtube.com:F1:P30 - Compare videos by view count and filter by 15+ minute duration",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F2:P7 - Extract video metadata
    metadata_complete = all([
        all(v.title for v in youtube_videos.videos if v.url),
        all(v.channel_name for v in youtube_videos.videos if v.url),
        all(v.view_count for v in youtube_videos.videos if v.url),
        all(v.duration for v in youtube_videos.videos if v.url),
        all(v.url for v in youtube_videos.videos if v.title)
    ])
    evaluator.add_custom_node(
        result=bool(metadata_complete and video_count >= 2),
        id="youtube_perception_metadata",
        desc="[Perception Node] youtube.com:F2:P7 - Extract complete video metadata (title, channel, views, duration, URL)",
        parent=youtube_node,
        critical=False
    )

    # URLs validation
    youtube_urls_ok = all(looks_like_url(v.url, 'youtube.com') or looks_like_url(v.url, 'youtu.be') for v in youtube_videos.videos if v.title)
    evaluator.add_custom_node(
        result=bool(youtube_urls_ok and video_count >= 2),
        id="youtube_urls_present",
        desc="YouTube video URLs are provided for 2 videos",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
