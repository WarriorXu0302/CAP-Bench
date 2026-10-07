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
TASK_ID = "task-c185a5"
TASK_DESCRIPTION = "I recently came across a highly-rated classic film on Letterboxd and am interested in knowing if its Criterion restoration is worth purchasing for my collection.\n\nFirst, please search for 'Seven Samurai' on Letterboxd. Record its community average rating, the number of ratings, and identify the top 3 most-liked reviews sorted by 'Popular'. For each of these reviews, extract the author, the number of likes, and a summary of the comment (first 100 words).\n\nNext, go to the official Criterion website and search for the film. View the restoration details and record the available format options (DVD/Blu-ray/4K), the price for each format, and the Spine number. Expand 'Special Features' to view the complete list of special supplements (e.g., audio commentaries, documentaries, interviews), and record the Aspect Ratio, Audio Format, and Subtitle Languages from the 'Technical Information' section.\n\nThen, search for the film on IMDb, navigate to its detail page, click to switch to the 'Technical Specs' tab, and extract the Runtime, Aspect Ratio, and Sound Mix.\n\nAfterward, search for the film on Metacritic and Rotten Tomatoes separately. Record Metacritic's Metascore and the number of reviews, as well as Rotten Tomatoes' Tomatometer percentage, the number of reviews, and whether it has a 'Certified Fresh' indicator.\n\nFinally, on YouTube, search for 'Seven Samurai 4K restoration comparison'. Sort the results by view count, filter for videos with over 50,000 views, and identify the video with the highest view count among them. Record its video title, channel name, and view count.\n\nPlease summarize and output the following: Letterboxd rating, number of ratings, 3 most-liked reviews (author/likes/summary) along with their respective links, Criterion format prices and Spine number, special supplements list, technical specifications (aspect ratio/audio/subtitles) and the Criterion detail page link, IMDb runtime and technical specifications, Metacritic Metascore and number of reviews, Rotten Tomatoes score and 'Certified Fresh' status, and YouTube comparison video title/channel name/view count along with its link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LetterboxdInfo(BaseModel):
    """Letterboxd rating and reviews data"""
    average_rating: Optional[str] = None
    num_ratings: Optional[str] = None
    review_1_author: Optional[str] = None
    review_1_likes: Optional[str] = None
    review_1_summary: Optional[str] = None
    review_1_link: Optional[str] = None
    review_2_author: Optional[str] = None
    review_2_likes: Optional[str] = None
    review_2_summary: Optional[str] = None
    review_2_link: Optional[str] = None
    review_3_author: Optional[str] = None
    review_3_likes: Optional[str] = None
    review_3_summary: Optional[str] = None
    review_3_link: Optional[str] = None


class CriterionInfo(BaseModel):
    """Criterion format, pricing, and technical details"""
    spine_number: Optional[str] = None
    dvd_price: Optional[str] = None
    bluray_price: Optional[str] = None
    four_k_price: Optional[str] = None
    special_features: Optional[str] = None
    aspect_ratio: Optional[str] = None
    audio_format: Optional[str] = None
    subtitle_languages: Optional[str] = None
    detail_page_link: Optional[str] = None


class IMDbInfo(BaseModel):
    """IMDb technical specifications"""
    runtime: Optional[str] = None
    aspect_ratio: Optional[str] = None
    sound_mix: Optional[str] = None


class MetacriticInfo(BaseModel):
    """Metacritic score and reviews"""
    metascore: Optional[str] = None
    num_reviews: Optional[str] = None


class RottenTomatoesInfo(BaseModel):
    """Rotten Tomatoes score and certification"""
    tomatometer: Optional[str] = None
    num_reviews: Optional[str] = None
    certified_fresh: Optional[str] = None


class YouTubeInfo(BaseModel):
    """YouTube comparison video details"""
    video_title: Optional[str] = None
    channel_name: Optional[str] = None
    view_count: Optional[str] = None
    video_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_letterboxd() -> str:
    return """
Extract the Letterboxd information for Seven Samurai from the answer:
- average_rating: the community average rating
- num_ratings: the number of ratings
- For the top 3 most-liked reviews sorted by 'Popular':
  - review_1_author, review_1_likes, review_1_summary, review_1_link
  - review_2_author, review_2_likes, review_2_summary, review_2_link
  - review_3_author, review_3_likes, review_3_summary, review_3_link

Set any missing field to null.
"""


def prompt_extract_criterion() -> str:
    return """
Extract the Criterion information from the answer:
- spine_number: the Spine number
- dvd_price: DVD format price if available
- bluray_price: Blu-ray format price if available
- four_k_price: 4K format price if available
- special_features: complete list of special supplements
- aspect_ratio: aspect ratio from technical information
- audio_format: audio format from technical information
- subtitle_languages: subtitle languages from technical information
- detail_page_link: link to the Criterion detail page

Set any missing field to null.
"""


def prompt_extract_imdb() -> str:
    return """
Extract IMDb technical specifications from the answer:
- runtime: the runtime
- aspect_ratio: the aspect ratio
- sound_mix: the sound mix

Set any missing field to null.
"""


def prompt_extract_metacritic() -> str:
    return """
Extract Metacritic information from the answer:
- metascore: the Metascore value
- num_reviews: the number of reviews

Set any missing field to null.
"""


def prompt_extract_rottentomatoes() -> str:
    return """
Extract Rotten Tomatoes information from the answer:
- tomatometer: the Tomatometer percentage
- num_reviews: the number of reviews
- certified_fresh: whether it has 'Certified Fresh' status (true/false/null)

Set any missing field to null.
"""


def prompt_extract_youtube() -> str:
    return """
Extract YouTube video information from the answer:
- video_title: the title of the highest view count video
- channel_name: the channel name
- view_count: the view count
- video_link: the link to the video

Set any missing field to null.
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


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Handle formatted numbers like "1,234" or "1.2k"
    text_clean = text.replace(',', '').lower()
    if 'k' in text_clean:
        m = re.search(r'(\d+(?:\.\d+)?)\s*k', text_clean)
        if m:
            try:
                return int(float(m.group(1)) * 1000)
            except Exception:
                return None
    m = re.search(r'(\d+)', text_clean)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_float(text)
    return val is not None and 0 <= val <= 5


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['$', 'usd', 'price']) and contains_digits(text)


def looks_like_spine_number(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) or has_any_ci(text, ['#', 'spine'])


def looks_like_percentage(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_int(text)
    return val is not None and 0 <= val <= 100


def looks_like_metascore(text: Optional[str]) -> bool:
    if not text:
        return False
    val = extract_int(text)
    return val is not None and 0 <= val <= 100


def is_yes_no_bool(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    return t in ['yes', 'no', 'true', 'false']


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
    letterboxd_info = await evaluator.extract(
        prompt=prompt_extract_letterboxd(),
        template_class=LetterboxdInfo,
        extraction_name="letterboxd_info"
    )

    criterion_info = await evaluator.extract(
        prompt=prompt_extract_criterion(),
        template_class=CriterionInfo,
        extraction_name="criterion_info"
    )

    imdb_info = await evaluator.extract(
        prompt=prompt_extract_imdb(),
        template_class=IMDbInfo,
        extraction_name="imdb_info"
    )

    metacritic_info = await evaluator.extract(
        prompt=prompt_extract_metacritic(),
        template_class=MetacriticInfo,
        extraction_name="metacritic_info"
    )

    rottentomatoes_info = await evaluator.extract(
        prompt=prompt_extract_rottentomatoes(),
        template_class=RottenTomatoesInfo,
        extraction_name="rottentomatoes_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube(),
        template_class=YouTubeInfo,
        extraction_name="youtube_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Letterboxd section
    letterboxd_node = evaluator.add_sequential(
        id="letterboxd_section",
        desc="Letterboxd rating and reviews for Seven Samurai",
        parent=root,
        critical=False
    )

    # [Action Node] letterboxd.com:F1:A5 - Click into film detail
    letterboxd_search_ok = has_any_ci(answer, ['letterboxd', 'seven samurai'])
    evaluator.add_custom_node(
        result=bool(letterboxd_search_ok),
        id="letterboxd_click_detail",
        desc="[Action Node] letterboxd.com:F1:A5 - Search and click into Seven Samurai detail page on Letterboxd",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P6 - Extract rating data
    rating_ok = looks_like_rating(letterboxd_info.average_rating)
    num_ratings_ok = contains_digits(letterboxd_info.num_ratings)
    evaluator.add_custom_node(
        result=bool(rating_ok and num_ratings_ok),
        id="letterboxd_rating_data",
        desc="[Perception Node] letterboxd.com:F3:P6 - Extract community average rating and number of ratings",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F3:A16 - Browse reviews (may require pagination)
    reviews_mention_ok = has_any_ci(answer, ['review', 'popular'])
    evaluator.add_custom_node(
        result=bool(reviews_mention_ok),
        id="letterboxd_browse_reviews",
        desc="[Action Node] letterboxd.com:F3:A16 - Browse reviews sorted by Popular (may involve pagination)",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P7 - Extract review metadata
    review_1_ok = bool(letterboxd_info.review_1_author and letterboxd_info.review_1_likes and letterboxd_info.review_1_summary)
    review_2_ok = bool(letterboxd_info.review_2_author and letterboxd_info.review_2_likes and letterboxd_info.review_2_summary)
    review_3_ok = bool(letterboxd_info.review_3_author and letterboxd_info.review_3_likes and letterboxd_info.review_3_summary)
    all_reviews_ok = review_1_ok and review_2_ok and review_3_ok
    evaluator.add_custom_node(
        result=bool(all_reviews_ok),
        id="letterboxd_review_metadata",
        desc="[Perception Node] letterboxd.com:F3:P7 - Extract author, likes, and summary for top 3 most-liked reviews",
        parent=letterboxd_node,
        critical=False
    )

    # 3.2 Criterion section
    criterion_node = evaluator.add_sequential(
        id="criterion_section",
        desc="Criterion restoration details and technical specifications",
        parent=root,
        critical=False
    )

    # [Action Node] criterion.com:F1:A9 - Click into film detail
    criterion_search_ok = has_any_ci(answer, ['criterion', 'seven samurai'])
    evaluator.add_custom_node(
        result=bool(criterion_search_ok),
        id="criterion_click_detail",
        desc="[Action Node] criterion.com:F1:A9 - Search and click into Seven Samurai detail page on Criterion",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F2:A13 - Switch format tabs
    format_mention_ok = has_any_ci(answer, ['dvd', 'blu-ray', 'bluray', '4k', 'format'])
    evaluator.add_custom_node(
        result=bool(format_mention_ok),
        id="criterion_format_tabs",
        desc="[Action Node] criterion.com:F2:A13 - Switch between format tabs (DVD/Blu-ray/4K) to view prices",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P5 - Extract format availability and pricing
    spine_ok = looks_like_spine_number(criterion_info.spine_number)
    dvd_price_ok = looks_like_price(criterion_info.dvd_price) if criterion_info.dvd_price else True
    bluray_price_ok = looks_like_price(criterion_info.bluray_price) if criterion_info.bluray_price else True
    four_k_price_ok = looks_like_price(criterion_info.four_k_price) if criterion_info.four_k_price else True
    has_some_price = bool(criterion_info.dvd_price or criterion_info.bluray_price or criterion_info.four_k_price)
    evaluator.add_custom_node(
        result=bool(spine_ok and has_some_price),
        id="criterion_format_pricing",
        desc="[Perception Node] criterion.com:F2:P5 - Extract Spine number and format pricing (DVD/Blu-ray/4K)",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F2:A12 - Expand Special Features panel
    special_features_mention_ok = has_any_ci(answer, ['special features', 'supplements', 'commentary', 'documentary'])
    evaluator.add_custom_node(
        result=bool(special_features_mention_ok),
        id="criterion_expand_special_features",
        desc="[Action Node] criterion.com:F2:A12 - Expand the Special Features collapsible panel",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P3 - Extract special supplements
    special_features_ok = bool(criterion_info.special_features and len(criterion_info.special_features.strip()) > 20)
    evaluator.add_custom_node(
        result=bool(special_features_ok),
        id="criterion_special_supplements",
        desc="[Perception Node] criterion.com:F2:P3 - Extract complete list of special supplements",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P4 - Extract technical specifications
    aspect_ok = bool(criterion_info.aspect_ratio and contains_digits(criterion_info.aspect_ratio))
    audio_ok = bool(criterion_info.audio_format and len(criterion_info.audio_format.strip()) > 2)
    subtitle_ok = bool(criterion_info.subtitle_languages and len(criterion_info.subtitle_languages.strip()) > 2)
    tech_specs_ok = aspect_ok and audio_ok and subtitle_ok
    evaluator.add_custom_node(
        result=bool(tech_specs_ok),
        id="criterion_tech_specs",
        desc="[Perception Node] criterion.com:F2:P4 - Extract technical specifications (aspect ratio, audio format, subtitle languages)",
        parent=criterion_node,
        critical=False
    )

    # 3.3 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb technical specifications",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F1:A26 - Click into film detail
    imdb_search_ok = has_any_ci(answer, ['imdb', 'seven samurai'])
    evaluator.add_custom_node(
        result=bool(imdb_search_ok),
        id="imdb_click_detail",
        desc="[Action Node] imdb.com:F1:A26 - Search and click into Seven Samurai detail page on IMDb",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A7 - Switch to Technical Specs tab
    tech_specs_tab_ok = has_any_ci(answer, ['technical specs', 'technical specifications', 'tech specs'])
    evaluator.add_custom_node(
        result=bool(tech_specs_tab_ok),
        id="imdb_switch_tech_specs_tab",
        desc="[Action Node] imdb.com:F3:A7 - Switch to the Technical Specs tab",
        parent=imdb_node,
        critical=False
    )

    # Extract IMDb technical data
    runtime_ok = bool(imdb_info.runtime and contains_digits(imdb_info.runtime))
    imdb_aspect_ok = bool(imdb_info.aspect_ratio and contains_digits(imdb_info.aspect_ratio))
    sound_mix_ok = bool(imdb_info.sound_mix and len(imdb_info.sound_mix.strip()) > 2)
    imdb_tech_ok = runtime_ok and imdb_aspect_ok and sound_mix_ok
    evaluator.add_custom_node(
        result=bool(imdb_tech_ok),
        id="imdb_tech_data",
        desc="Extract runtime, aspect ratio, and sound mix from IMDb Technical Specs",
        parent=imdb_node,
        critical=False
    )

    # 3.4 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic score and reviews",
        parent=root,
        critical=False
    )

    # [Action Node] metacritic.com:F1:A14 - Click into film detail
    metacritic_search_ok = has_any_ci(answer, ['metacritic', 'seven samurai'])
    evaluator.add_custom_node(
        result=bool(metacritic_search_ok),
        id="metacritic_click_detail",
        desc="[Action Node] metacritic.com:F1:A14 - Search and click into Seven Samurai detail page on Metacritic",
        parent=metacritic_node,
        critical=False
    )

    # [Perception Node] metacritic.com:F4:P1 - Extract Metascore
    metascore_ok = looks_like_metascore(metacritic_info.metascore)
    metacritic_reviews_ok = contains_digits(metacritic_info.num_reviews)
    evaluator.add_custom_node(
        result=bool(metascore_ok and metacritic_reviews_ok),
        id="metacritic_score_data",
        desc="[Perception Node] metacritic.com:F4:P1 - Extract Metascore and number of reviews",
        parent=metacritic_node,
        critical=False
    )

    # 3.5 Rotten Tomatoes section
    rt_node = evaluator.add_sequential(
        id="rottentomatoes_section",
        desc="Rotten Tomatoes score and certification",
        parent=root,
        critical=False
    )

    # [Action Node] rottentomatoes.com:F1:A47 - Click into film detail
    rt_search_ok = has_any_ci(answer, ['rotten tomatoes', 'seven samurai'])
    evaluator.add_custom_node(
        result=bool(rt_search_ok),
        id="rt_click_detail",
        desc="[Action Node] rottentomatoes.com:F1:A47 - Search and click into Seven Samurai detail page on Rotten Tomatoes",
        parent=rt_node,
        critical=False
    )

    # Extract Rotten Tomatoes data
    tomatometer_ok = looks_like_percentage(rottentomatoes_info.tomatometer)
    rt_reviews_ok = contains_digits(rottentomatoes_info.num_reviews)
    evaluator.add_custom_node(
        result=bool(tomatometer_ok and rt_reviews_ok),
        id="rt_tomatometer_data",
        desc="Extract Tomatometer percentage and number of reviews",
        parent=rt_node,
        critical=False
    )

    # [Perception Node] rottentomatoes.com:F4:P2 - Identify Certified Fresh status
    certified_fresh_ok = bool(rottentomatoes_info.certified_fresh and len(rottentomatoes_info.certified_fresh.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(certified_fresh_ok),
        id="rt_certified_fresh",
        desc="[Perception Node] rottentomatoes.com:F4:P2 - Identify Certified Fresh status",
        parent=rt_node,
        critical=False
    )

    # 3.6 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube comparison video search and filtering",
        parent=root,
        critical=False
    )

    # YouTube search mention
    youtube_search_ok = has_any_ci(answer, ['youtube', 'restoration comparison', '4k'])
    evaluator.add_custom_node(
        result=bool(youtube_search_ok),
        id="youtube_search_restoration",
        desc="Search for 'Seven Samurai 4K restoration comparison' on YouTube",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Apply filters (sort by view count, filter >50k views)
    filter_mention_ok = has_any_ci(answer, ['view count', 'sort', 'filter', '50,000', '50000', '50k'])
    evaluator.add_custom_node(
        result=bool(filter_mention_ok),
        id="youtube_apply_filters",
        desc="[Action Node] youtube.com:F9:A4 - Sort by view count and filter for videos with over 50,000 views",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click video card
    video_title_ok = bool(youtube_info.video_title and len(youtube_info.video_title.strip()) > 5)
    evaluator.add_custom_node(
        result=bool(video_title_ok),
        id="youtube_click_video",
        desc="[Action Node] youtube.com:F1:A22 - Click on the highest view count video card",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Extract video metadata
    channel_ok = bool(youtube_info.channel_name and len(youtube_info.channel_name.strip()) > 0)
    view_count_num = extract_int(youtube_info.view_count)
    view_count_ok = view_count_num is not None and view_count_num >= 50000
    video_metadata_ok = video_title_ok and channel_ok and view_count_ok
    evaluator.add_custom_node(
        result=bool(video_metadata_ok),
        id="youtube_video_metadata",
        desc="[Perception Node] youtube.com:F1:P4 - Extract video title, channel name, and view count (must be >50k)",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
