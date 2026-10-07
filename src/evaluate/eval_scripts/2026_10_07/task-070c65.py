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
TASK_ID = "task-070c65"
TASK_DESCRIPTION = 'I’m preparing teaching materials for an Ancient Greek mythology art workshop. First, go to the Getty Museum website and find up to 5 Ancient Greek mythology-themed works that are currently on display (sculptures, ceramics, or reliefs are all acceptable). For each work, record its title, date, material, exhibition name, and the mythological figure or theme mentioned in its description (e.g., Athena, Heracles, the Trojan War, etc.). If there are fewer than 5 relevant works in Current exhibitions, supplement from the Getty collection database with additional Ancient Greek mythology-related works, and clearly mark which ones are from Current exhibitions and which are from the collection database.\n\nThen go to Bilibili and YouTube, and use the names of these mythological figures as search keywords for educational/explanatory videos:\n- On Bilibili: filter within the Knowledge category, find one video that is at least 8 minutes long with the highest view count.\n- On YouTube: filter for videos at least 8 minutes long, sort by view count, and find the one with the highest views.\n\nFinally, output for each artwork:\n- Artwork title  \n- Date  \n- Material  \n- Mythological theme  \n- Getty artwork detail page link  \n- Bilibili video title and link  \n- Bilibili video duration and view count  \n- YouTube video title and link  \n- YouTube video duration and view count'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GettyArtwork(BaseModel):
    """Single Getty artwork information"""
    title: Optional[str] = None
    date: Optional[str] = None
    material: Optional[str] = None
    mythological_theme: Optional[str] = None
    getty_link: Optional[str] = None
    is_current_exhibition: Optional[bool] = None


class BilibiliVideo(BaseModel):
    """Bilibili video information"""
    title: Optional[str] = None
    link: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[str] = None


class YouTubeVideo(BaseModel):
    """YouTube video information"""
    title: Optional[str] = None
    link: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[str] = None


class ArtworkWithVideos(BaseModel):
    """Complete information for one artwork with associated videos"""
    artwork: Optional[GettyArtwork] = None
    bilibili_video: Optional[BilibiliVideo] = None
    youtube_video: Optional[YouTubeVideo] = None


class AllArtworks(BaseModel):
    """All artworks extracted from the answer"""
    artworks: List[ArtworkWithVideos] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_artworks() -> str:
    return """
Extract all Ancient Greek mythology-themed artworks from the answer, along with their associated Bilibili and YouTube videos.

For each artwork, extract:
- title: artwork title
- date: date or time period
- material: material description (sculpture, ceramic, relief, etc.)
- mythological_theme: the mythological figure or theme mentioned (e.g., Athena, Heracles, Trojan War)
- getty_link: Getty Museum detail page URL
- is_current_exhibition: true if explicitly marked as from Current exhibitions, false if from collection database, null if unclear
- bilibili_video: {title, link, duration, view_count}
- youtube_video: {title, link, duration, view_count}

Return all artworks found in the answer. If any field is missing, set it to null.
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


def looks_like_getty_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return 'getty' in url.lower() and ('http://' in url or 'https://' in url)


def looks_like_bilibili_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return 'bilibili' in url.lower() or 'b23.tv' in url.lower()


def looks_like_youtube_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return 'youtube' in url.lower() or 'youtu.be' in url.lower()


def parse_duration_minutes(duration_text: Optional[str]) -> Optional[float]:
    """Parse duration text and return minutes as float"""
    if not duration_text:
        return None

    # Try to extract mm:ss or hh:mm:ss format
    patterns = [
        r'(\d+):(\d+):(\d+)',  # hh:mm:ss
        r'(\d+):(\d+)',         # mm:ss
        r'(\d+)\s*min',         # "X min"
        r'(\d+)\s*分',          # "X分"
    ]

    for pattern in patterns:
        match = re.search(pattern, duration_text)
        if match:
            groups = match.groups()
            if len(groups) == 3:  # hh:mm:ss
                return int(groups[0]) * 60 + int(groups[1]) + int(groups[2]) / 60
            elif len(groups) == 2:  # mm:ss
                return int(groups[0]) + int(groups[1]) / 60
            elif len(groups) == 1:  # minutes
                return float(groups[0])

    return None


def looks_like_view_count(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept numbers with potential units like K, M, 万, 亿, etc.
    return contains_digits(text)


def mentions_greek_mythology(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['greek', 'mythology', 'athena', 'heracles', 'hercules', 'zeus',
                'apollo', 'trojan', 'odysseus', 'poseidon', 'hera', 'artemis',
                '希腊', '神话', '雅典娜', '赫拉克勒斯']
    return has_any_ci(text, keywords)


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
    all_artworks_info = await evaluator.extract(
        prompt=prompt_extract_all_artworks(),
        template_class=AllArtworks,
        extraction_name="all_artworks_with_videos"
    )

    artworks = all_artworks_info.artworks if all_artworks_info else []

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Getty Museum section
    getty_node = evaluator.add_sequential(
        id="getty_museum_section",
        desc="Getty Museum - Find Ancient Greek mythology artworks",
        parent=root,
        critical=False
    )

    # [Action Node] getty.edu:F2:A1 - Switch to Current exhibitions tab
    mentions_current = has_any_ci(answer, ['current', 'exhibition', 'on display', '正在展出'])
    has_current_marked = any(
        artwork.artwork and artwork.artwork.is_current_exhibition == True
        for artwork in artworks
    )
    evaluator.add_custom_node(
        result=bool(mentions_current or has_current_marked),
        id="getty_action_current_tab",
        desc="[Action Node] getty.edu:F2:A1 - Switch to Current exhibitions tab to find currently displayed works",
        parent=getty_node,
        critical=False
    )

    # [Action Node] getty.edu:F2:A7 - Click artwork cards to enter details
    has_valid_links = any(
        artwork.artwork and looks_like_getty_url(artwork.artwork.getty_link)
        for artwork in artworks
    )
    evaluator.add_custom_node(
        result=bool(has_valid_links),
        id="getty_action_click_cards",
        desc="[Action Node] getty.edu:F2:A7 - Click artwork cards to view detail pages",
        parent=getty_node,
        critical=False
    )

    # [Action Node] getty.edu:F2:A13 - Select artworks in Selected Works area
    has_titles = any(
        artwork.artwork and artwork.artwork.title
        for artwork in artworks
    )
    evaluator.add_custom_node(
        result=bool(has_titles and len(artworks) > 0),
        id="getty_action_select_works",
        desc="[Action Node] getty.edu:F2:A13 - Select relevant Ancient Greek mythology artworks",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F1:P1 - Identify object types from images
    has_materials = any(
        artwork.artwork and artwork.artwork.material and
        has_any_ci(artwork.artwork.material, ['sculpture', 'ceramic', 'relief', 'pottery', 'terracotta', 'marble', 'bronze', '雕塑', '陶器', '浮雕'])
        for artwork in artworks
    )
    evaluator.add_custom_node(
        result=bool(has_materials),
        id="getty_perception_object_types",
        desc="[Perception Node] getty.edu:F1:P1 - Identify object types (sculpture, ceramic, relief) from artwork images",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F2:P5 - Recognize Current exhibition status
    clearly_marked = any(
        artwork.artwork and artwork.artwork.is_current_exhibition is not None
        for artwork in artworks
    )
    evaluator.add_custom_node(
        result=bool(clearly_marked or mentions_current),
        id="getty_perception_current_status",
        desc="[Perception Node] getty.edu:F2:P5 - Recognize and distinguish Current exhibitions from collection database",
        parent=getty_node,
        critical=False
    )

    # Check completeness of Getty information (o1-o5)
    complete_artworks = [
        artwork for artwork in artworks
        if artwork.artwork and
           artwork.artwork.title and
           artwork.artwork.date and
           artwork.artwork.material and
           artwork.artwork.mythological_theme and
           looks_like_getty_url(artwork.artwork.getty_link)
    ]

    evaluator.add_custom_node(
        result=bool(len(complete_artworks) >= 3),
        id="getty_completeness",
        desc="At least 3 artworks with complete information (title, date, material, theme, link)",
        parent=getty_node,
        critical=False
    )

    # 3.2 Bilibili section
    bilibili_node = evaluator.add_sequential(
        id="bilibili_section",
        desc="Bilibili - Search for mythology educational videos",
        parent=root,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A48 - Expand duration filter panel
    bilibili_videos = [artwork.bilibili_video for artwork in artworks if artwork.bilibili_video]
    has_duration_8min = any(
        video and video.duration and parse_duration_minutes(video.duration) and parse_duration_minutes(video.duration) >= 8
        for video in bilibili_videos
    )
    evaluator.add_custom_node(
        result=bool(has_duration_8min or has_any_ci(answer, ['8分钟', '8 min', '8 minutes', '480'])),
        id="bilibili_action_duration_filter",
        desc="[Action Node] bilibili.com:F1:A48 - Expand and apply duration filter (8 minutes or longer)",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F1:A5 - Sort by view count
    mentions_sort = has_any_ci(answer, ['播放量', 'view count', 'most viewed', 'highest view', '最高'])
    has_view_counts = any(
        video and looks_like_view_count(video.view_count)
        for video in bilibili_videos
    )
    evaluator.add_custom_node(
        result=bool(mentions_sort or has_view_counts),
        id="bilibili_action_sort_views",
        desc="[Action Node] bilibili.com:F1:A5 - Sort search results by view count",
        parent=bilibili_node,
        critical=False
    )

    # [Action Node] bilibili.com:F3:A33 - Click video cards
    has_bilibili_links = any(
        video and looks_like_bilibili_url(video.link)
        for video in bilibili_videos
    )
    evaluator.add_custom_node(
        result=bool(has_bilibili_links),
        id="bilibili_action_click_cards",
        desc="[Action Node] bilibili.com:F3:A33 - Click filtered video cards to access details",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P30 - Recognize educational video thumbnails
    mentions_knowledge = has_any_ci(answer, ['知识', 'knowledge', 'educational', 'explanatory', '科普', '讲解'])
    has_video_titles = any(
        video and video.title and mentions_greek_mythology(video.title)
        for video in bilibili_videos
    )
    evaluator.add_custom_node(
        result=bool(mentions_knowledge or has_video_titles),
        id="bilibili_perception_educational",
        desc="[Perception Node] bilibili.com:F1:P30 - Recognize educational/explanatory content from thumbnails and titles",
        parent=bilibili_node,
        critical=False
    )

    # [Perception Node] bilibili.com:F1:P31 - Identify and skip ads
    evaluator.add_custom_node(
        result=bool(has_bilibili_links),
        id="bilibili_perception_skip_ads",
        desc="[Perception Node] bilibili.com:F1:P31 - Identify and skip advertisement videos",
        parent=bilibili_node,
        critical=False
    )

    # Check completeness of Bilibili information (o6-o9)
    complete_bilibili = [
        video for video in bilibili_videos
        if video and video.title and looks_like_bilibili_url(video.link) and
           video.duration and video.view_count
    ]

    evaluator.add_custom_node(
        result=bool(len(complete_bilibili) >= 2),
        id="bilibili_completeness",
        desc="At least 2 Bilibili videos with complete information (title, link, duration, view count)",
        parent=bilibili_node,
        critical=False
    )

    # 3.3 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube - Search for mythology educational videos",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Duration filter
    youtube_videos = [artwork.youtube_video for artwork in artworks if artwork.youtube_video]
    youtube_duration_8min = any(
        video and video.duration and parse_duration_minutes(video.duration) and parse_duration_minutes(video.duration) >= 8
        for video in youtube_videos
    )
    evaluator.add_custom_node(
        result=bool(youtube_duration_8min or has_any_ci(answer, ['8 min', '8 minutes', '480'])),
        id="youtube_action_duration_filter",
        desc="[Action Node] youtube.com:F9:A4 - Apply duration filter (longer than 8 minutes)",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click video from sorted results
    has_youtube_links = any(
        video and looks_like_youtube_url(video.link)
        for video in youtube_videos
    )
    evaluator.add_custom_node(
        result=bool(has_youtube_links),
        id="youtube_action_click_video",
        desc="[Action Node] youtube.com:F1:A22 - Click video from view-count sorted results",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Recognize content relevance
    youtube_relevant = any(
        video and video.title and mentions_greek_mythology(video.title)
        for video in youtube_videos
    )
    evaluator.add_custom_node(
        result=bool(youtube_relevant),
        id="youtube_perception_relevance",
        desc="[Perception Node] youtube.com:F1:P4 - Identify videos relevant to the mythology theme",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P5 - Identify and skip ads
    evaluator.add_custom_node(
        result=bool(has_youtube_links),
        id="youtube_perception_skip_ads",
        desc="[Perception Node] youtube.com:F1:P5 - Identify and skip advertisement videos",
        parent=youtube_node,
        critical=False
    )

    # Check completeness of YouTube information (o10-o13)
    complete_youtube = [
        video for video in youtube_videos
        if video and video.title and looks_like_youtube_url(video.link) and
           video.duration and video.view_count
    ]

    evaluator.add_custom_node(
        result=bool(len(complete_youtube) >= 2),
        id="youtube_completeness",
        desc="At least 2 YouTube videos with complete information (title, link, duration, view count)",
        parent=youtube_node,
        critical=False
    )

    # 3.4 Overall integration check
    evaluator.add_custom_node(
        result=bool(len(artworks) >= 3 and len(complete_artworks) >= 2),
        id="overall_integration",
        desc="Successfully integrated Getty artworks with video recommendations",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
