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
TASK_ID = "task-e65fd1"
TASK_DESCRIPTION = "I am preparing a music documentary about the 90s indie rock revival and wish to evaluate the market potential and production feasibility of several candidate themes.\n\nFirst, search IMDb and Letterboxd for existing documentaries related to 90s indie rock. Identify 3 documentaries with a rating of 7.0 or higher, and then analyze audience comments to understand their primary expectations regarding angles and any dissatisfactions with current productions.\n\nNext, go to Spotify to look up the bands or artists prominently featured in these three documentaries. Record their current monthly listeners and their top 3 most popular songs to assess their market popularity.\n\nFinally, on Discogs, research the representative albums of these artists to determine the number of release versions, the main record labels, and the counts of collectors and wantlisters. This information will help gauge the complexity of music copyright acquisition.\n\nFor each candidate theme, output the following: Documentary Title, IMDb/Letterboxd Rating and Link, Main Audience Expectations and Dissatisfactions from Comments, Core Artist Name, Spotify Monthly Listeners, Top 3 Popular Tracks, Representative Album Title, Discogs Number of Versions, Main Record Labels, Number of Collectors/Wantlisters, and your comprehensive assessment of the theme's market potential and production difficulty."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DocumentaryInfo(BaseModel):
    """Documentary information extracted from the answer"""
    title: Optional[str] = None
    rating: Optional[str] = None
    link: Optional[str] = None
    source: Optional[str] = None


class AudienceInsights(BaseModel):
    """Audience expectations and dissatisfactions extracted from the answer"""
    expectations: Optional[str] = None
    dissatisfactions: Optional[str] = None


class ArtistInfo(BaseModel):
    """Artist/band information extracted from the answer"""
    name: Optional[str] = None
    monthly_listeners: Optional[str] = None
    top_tracks: Optional[List[str]] = Field(default_factory=list)


class AlbumInfo(BaseModel):
    """Album information from Discogs extracted from the answer"""
    title: Optional[str] = None
    num_versions: Optional[str] = None
    labels: Optional[str] = None
    collectors: Optional[str] = None
    wantlisters: Optional[str] = None


class ThemeAssessment(BaseModel):
    """Comprehensive assessment of theme extracted from the answer"""
    market_potential: Optional[str] = None
    production_difficulty: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_documentaries() -> str:
    return """
Extract information about the documentaries found on IMDb and Letterboxd from the answer.

For each documentary, return:
- title: the documentary title
- rating: the rating (as stated, e.g., "7.5", "8.2/10", "4.1/5")
- link: the IMDb or Letterboxd URL if provided
- source: which platform ("IMDb" or "Letterboxd")

Return a list of documentaries. If none found, return empty list.
"""


def prompt_extract_audience_insights() -> str:
    return """
Extract the audience expectations and dissatisfactions from the documentary comments mentioned in the answer.

Return:
- expectations: summary of what audiences expect or want to see in documentaries about this topic
- dissatisfactions: summary of what audiences criticize or find lacking in existing documentaries

If not present, set fields to null.
"""


def prompt_extract_artists() -> str:
    return """
Extract information about the artists/bands mentioned from the Spotify lookups in the answer.

For each artist, return:
- name: artist or band name
- monthly_listeners: the monthly listener count as stated
- top_tracks: list of the top 3 popular songs mentioned

Return a list of artists. If none found, return empty list.
"""


def prompt_extract_albums() -> str:
    return """
Extract information about the representative albums from Discogs mentioned in the answer.

For each album, return:
- title: album title
- num_versions: number of release versions
- labels: main record label(s)
- collectors: number of collectors
- wantlisters: number of people who want it

Return a list of albums. If none found, return empty list.
"""


def prompt_extract_assessment() -> str:
    return """
Extract the comprehensive assessment of the theme's market potential and production difficulty from the answer.

Return:
- market_potential: assessment of market potential
- production_difficulty: assessment of production difficulty/complexity

If not present, set fields to null.
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


def extract_rating_value(rating_text: Optional[str]) -> Optional[float]:
    if not rating_text:
        return None
    m = re.findall(r'(\d+\.?\d*)', rating_text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def is_valid_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def looks_like_listener_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['listener', 'monthly']) or re.search(r'\d{3,}', text))


def looks_like_version_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['version', 'release']) or re.match(r'^\d+$', text.strip()))


def looks_like_collector_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['collector', 'have']) or re.match(r'^\d+$', text.strip()))


def looks_like_wantlist_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['want']) or re.match(r'^\d+$', text.strip()))


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
    # We'll use a simple approach: extract as lists/dicts and check presence
    # For this task, we'll do targeted extractions per section

    # Extract documentary info
    class DocumentaryList(BaseModel):
        documentaries: List[DocumentaryInfo] = Field(default_factory=list)

    doc_list = await evaluator.extract(
        prompt=prompt_extract_documentaries(),
        template_class=DocumentaryList,
        extraction_name="documentaries"
    )

    audience_insights = await evaluator.extract(
        prompt=prompt_extract_audience_insights(),
        template_class=AudienceInsights,
        extraction_name="audience_insights"
    )

    class ArtistList(BaseModel):
        artists: List[ArtistInfo] = Field(default_factory=list)

    artist_list = await evaluator.extract(
        prompt=prompt_extract_artists(),
        template_class=ArtistList,
        extraction_name="artists"
    )

    class AlbumList(BaseModel):
        albums: List[AlbumInfo] = Field(default_factory=list)

    album_list = await evaluator.extract(
        prompt=prompt_extract_albums(),
        template_class=AlbumList,
        extraction_name="albums"
    )

    assessment = await evaluator.extract(
        prompt=prompt_extract_assessment(),
        template_class=ThemeAssessment,
        extraction_name="assessment"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 IMDb Section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb documentary search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F1:A1 - Search type dropdown
    imdb_search_type_ok = has_any_ci(answer, ['imdb']) and has_any_ci(answer, ['documentar'])
    evaluator.add_custom_node(
        result=bool(imdb_search_type_ok),
        id="imdb_search_type",
        desc="[Action Node] imdb.com:F1:A1 - Search for documentaries using appropriate type selection",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F1:A10 - Pagination
    # Check if multiple documentaries found (suggests pagination may have been used)
    imdb_docs = [d for d in doc_list.documentaries if d.source and ci_contains(d.source, 'imdb')]
    pagination_ok = len(imdb_docs) >= 2
    evaluator.add_custom_node(
        result=bool(pagination_ok),
        id="imdb_pagination",
        desc="[Action Node] imdb.com:F1:A10 - Navigate through search results to find multiple qualifying documentaries",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A26 - Click card to enter details
    imdb_detail_ok = any(d.link and is_valid_url(d.link) for d in imdb_docs)
    evaluator.add_custom_node(
        result=bool(imdb_detail_ok),
        id="imdb_click_detail",
        desc="[Action Node] imdb.com:F3:A26 - Click documentary card to view full rating and reviews",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F3:P1 - Identify poster content
    imdb_theme_ok = has_any_ci(answer, ['90s', '1990', 'indie rock', 'independent rock'])
    evaluator.add_custom_node(
        result=bool(imdb_theme_ok),
        id="imdb_poster_perception",
        desc="[Perception Node] imdb.com:F3:P1 - Recognize 90s indie rock theme from documentary posters",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F4:A40 - Scroll to load more reviews
    review_depth_ok = has_any_ci(answer, ['comment', 'review', 'expectation', 'dissatisfaction'])
    evaluator.add_custom_node(
        result=bool(review_depth_ok),
        id="imdb_scroll_reviews",
        desc="[Action Node] imdb.com:F4:A40 - Scroll through reviews to gather sufficient audience feedback",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F4:P25 - Understand review content
    review_understanding_ok = (audience_insights.expectations or audience_insights.dissatisfactions)
    evaluator.add_custom_node(
        result=bool(review_understanding_ok),
        id="imdb_review_understanding",
        desc="[Perception Node] imdb.com:F4:P25 - Extract audience expectations and criticisms from review text",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Letterboxd Section
    letterboxd_node = evaluator.add_sequential(
        id="letterboxd_section",
        desc="Letterboxd documentary search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] letterboxd.com:F1:A1 - Decade filter
    letterboxd_decade_ok = has_any_ci(answer, ['letterboxd']) and has_any_ci(answer, ['90s', '1990'])
    evaluator.add_custom_node(
        result=bool(letterboxd_decade_ok),
        id="letterboxd_decade_filter",
        desc="[Action Node] letterboxd.com:F1:A1 - Apply 1990s decade filter to search",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F1:A2 - Genre filter
    letterboxd_genre_ok = has_any_ci(answer, ['letterboxd']) and has_any_ci(answer, ['documentar'])
    evaluator.add_custom_node(
        result=bool(letterboxd_genre_ok),
        id="letterboxd_genre_filter",
        desc="[Action Node] letterboxd.com:F1:A2 - Select Documentary genre filter",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F1:A5 - Click card to details
    letterboxd_docs = [d for d in doc_list.documentaries if d.source and ci_contains(d.source, 'letterboxd')]
    letterboxd_detail_ok = any(d.link and is_valid_url(d.link) for d in letterboxd_docs)
    evaluator.add_custom_node(
        result=bool(letterboxd_detail_ok),
        id="letterboxd_click_detail",
        desc="[Action Node] letterboxd.com:F1:A5 - Click film poster to view ratings and reviews",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P6 - Identify rating
    letterboxd_rating_ok = any(d.rating and contains_digits(d.rating) for d in letterboxd_docs)
    evaluator.add_custom_node(
        result=bool(letterboxd_rating_ok),
        id="letterboxd_rating_perception",
        desc="[Perception Node] letterboxd.com:F3:P6 - Extract average rating from film detail page",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P7 - Understand reviews
    letterboxd_review_ok = has_any_ci(answer, ['letterboxd']) and review_understanding_ok
    evaluator.add_custom_node(
        result=bool(letterboxd_review_ok),
        id="letterboxd_review_understanding",
        desc="[Perception Node] letterboxd.com:F3:P7 - Analyze review content for audience insights",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F3:A15 - Expand spoiler reviews
    spoiler_ok = has_any_ci(answer, ['review', 'comment'])
    evaluator.add_custom_node(
        result=bool(spoiler_ok),
        id="letterboxd_expand_spoilers",
        desc="[Action Node] letterboxd.com:F3:A15 - Expand spoiler-tagged reviews to read full content",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F3:A16 - Paginate reviews
    review_pagination_ok = has_any_ci(answer, ['expectation', 'dissatisfaction', 'audience'])
    evaluator.add_custom_node(
        result=bool(review_pagination_ok),
        id="letterboxd_review_pagination",
        desc="[Action Node] letterboxd.com:F3:A16 - Navigate through multiple pages of reviews",
        parent=letterboxd_node,
        critical=False
    )

    # 3.3 Spotify Section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify artist research",
        parent=root,
        critical=False
    )

    # [Perception Node] open.spotify.com:F1:P12 - Understand search results
    spotify_search_ok = has_any_ci(answer, ['spotify']) and len(artist_list.artists) > 0
    evaluator.add_custom_node(
        result=bool(spotify_search_ok),
        id="spotify_search_results",
        desc="[Perception Node] open.spotify.com:F1:P12 - Identify artist information in search results",
        parent=spotify_node,
        critical=False
    )

    # [Action Node] open.spotify.com:F3:A9 - Click artist card
    artist_links_ok = len(artist_list.artists) > 0 and any(a.name for a in artist_list.artists)
    evaluator.add_custom_node(
        result=bool(artist_links_ok),
        id="spotify_click_artist",
        desc="[Action Node] open.spotify.com:F3:A9 - Click artist card to view profile",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F3:P1 - Recognize artist cover
    artist_style_ok = has_any_ci(answer, ['indie rock', '90s']) and len(artist_list.artists) > 0
    evaluator.add_custom_node(
        result=bool(artist_style_ok),
        id="spotify_artist_cover",
        desc="[Perception Node] open.spotify.com:F3:P1 - Recognize 90s indie rock artists from cover images",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F3:P12 - Extract monthly listeners and top tracks
    listener_ok = any(a.monthly_listeners and looks_like_listener_count(a.monthly_listeners) for a in artist_list.artists)
    tracks_ok = any(a.top_tracks and len(a.top_tracks) >= 3 for a in artist_list.artists)
    evaluator.add_custom_node(
        result=bool(listener_ok and tracks_ok),
        id="spotify_extract_data",
        desc="[Perception Node] open.spotify.com:F3:P12 - Extract monthly listeners and top 3 popular tracks",
        parent=spotify_node,
        critical=False
    )

    # 3.4 Discogs Section
    discogs_node = evaluator.add_sequential(
        id="discogs_section",
        desc="Discogs album research",
        parent=root,
        critical=False
    )

    # [Action Node] discogs.com:F1:A7 - Click album card
    discogs_click_ok = has_any_ci(answer, ['discogs']) and len(album_list.albums) > 0
    evaluator.add_custom_node(
        result=bool(discogs_click_ok),
        id="discogs_click_album",
        desc="[Action Node] discogs.com:F1:A7 - Click album card to view details",
        parent=discogs_node,
        critical=False
    )

    # [Perception Node] discogs.com:F1:P1 - Recognize album cover
    album_match_ok = len(album_list.albums) > 0 and any(a.title for a in album_list.albums)
    evaluator.add_custom_node(
        result=bool(album_match_ok),
        id="discogs_album_cover",
        desc="[Perception Node] discogs.com:F1:P1 - Identify representative albums from cover images",
        parent=discogs_node,
        critical=False
    )

    # [Action Node] discogs.com:F2:A4 - Switch to Versions tab
    versions_tab_ok = any(a.num_versions and looks_like_version_count(a.num_versions) for a in album_list.albums)
    evaluator.add_custom_node(
        result=bool(versions_tab_ok),
        id="discogs_versions_tab",
        desc="[Action Node] discogs.com:F2:A4 - Navigate to Versions tab to view release variants",
        parent=discogs_node,
        critical=False
    )

    # [Perception Node] discogs.com:F2:P14 - Understand version list
    version_count_ok = any(a.num_versions and contains_digits(a.num_versions) for a in album_list.albums)
    evaluator.add_custom_node(
        result=bool(version_count_ok),
        id="discogs_version_count",
        desc="[Perception Node] discogs.com:F2:P14 - Extract total number of release versions",
        parent=discogs_node,
        critical=False
    )

    # [Perception Node] discogs.com:F2:P4 - Extract label info
    label_ok = any(a.labels for a in album_list.albums)
    evaluator.add_custom_node(
        result=bool(label_ok),
        id="discogs_label_info",
        desc="[Perception Node] discogs.com:F2:P4 - Identify main record label from album details",
        parent=discogs_node,
        critical=False
    )

    # [Action Node] discogs.com:F2:A10 - Expand companies panel
    companies_ok = has_any_ci(answer, ['label', 'record']) and label_ok
    evaluator.add_custom_node(
        result=bool(companies_ok),
        id="discogs_expand_companies",
        desc="[Action Node] discogs.com:F2:A10 - Expand companies/labels information panel",
        parent=discogs_node,
        critical=False
    )

    # [Perception Node] discogs.com:F8:P15 - Extract statistics
    collector_ok = any(a.collectors and looks_like_collector_count(a.collectors) for a in album_list.albums)
    wantlist_ok = any(a.wantlisters and looks_like_wantlist_count(a.wantlisters) for a in album_list.albums)
    evaluator.add_custom_node(
        result=bool(collector_ok and wantlist_ok),
        id="discogs_statistics",
        desc="[Perception Node] discogs.com:F8:P15 - Extract Have (collectors) and Want (wantlisters) counts",
        parent=discogs_node,
        critical=False
    )

    # 3.5 Overall output completeness checks
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Comprehensive output format",
        parent=root,
        critical=False
    )

    # Check for required 3 documentaries with rating >= 7.0
    qualified_docs = [d for d in doc_list.documentaries if d.rating and extract_rating_value(d.rating)]
    three_docs_ok = len(qualified_docs) >= 3
    evaluator.add_custom_node(
        result=bool(three_docs_ok),
        id="three_documentaries",
        desc="Identified at least 3 documentaries with ratings",
        parent=output_node,
        critical=False
    )

    # Check rating threshold (lenient: accept if mentioned or if values seem >= 7.0 or equivalent)
    rating_threshold_ok = any(
        extract_rating_value(d.rating) and
        (extract_rating_value(d.rating) >= 7.0 or
         (extract_rating_value(d.rating) >= 3.5 and has_any_ci(d.rating, ['/5', 'out of 5'])))
        for d in qualified_docs
    )
    evaluator.add_custom_node(
        result=bool(rating_threshold_ok),
        id="rating_threshold",
        desc="Documentaries meet 7.0+ rating threshold (or equivalent scale)",
        parent=output_node,
        critical=False
    )

    # Check links provided
    links_ok = any(d.link and is_valid_url(d.link) for d in doc_list.documentaries)
    evaluator.add_custom_node(
        result=bool(links_ok),
        id="documentary_links",
        desc="Provided valid IMDb/Letterboxd links",
        parent=output_node,
        critical=False
    )

    # Check audience analysis
    audience_analysis_ok = bool(audience_insights.expectations or audience_insights.dissatisfactions)
    evaluator.add_custom_node(
        result=bool(audience_analysis_ok),
        id="audience_analysis",
        desc="Analyzed audience expectations and dissatisfactions",
        parent=output_node,
        critical=False
    )

    # Check artist data
    artist_data_ok = len(artist_list.artists) > 0 and any(
        a.monthly_listeners and a.top_tracks and len(a.top_tracks) >= 3
        for a in artist_list.artists
    )
    evaluator.add_custom_node(
        result=bool(artist_data_ok),
        id="artist_data",
        desc="Provided artist names, monthly listeners, and top 3 tracks",
        parent=output_node,
        critical=False
    )

    # Check album copyright data
    copyright_data_ok = len(album_list.albums) > 0 and any(
        a.num_versions and a.labels and a.collectors and a.wantlisters
        for a in album_list.albums
    )
    evaluator.add_custom_node(
        result=bool(copyright_data_ok),
        id="copyright_data",
        desc="Provided version counts, labels, collectors, and wantlisters",
        parent=output_node,
        critical=False
    )

    # Check comprehensive assessment
    assessment_ok = bool(assessment.market_potential or assessment.production_difficulty)
    evaluator.add_custom_node(
        result=bool(assessment_ok),
        id="comprehensive_assessment",
        desc="Provided assessment of market potential and production difficulty",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
