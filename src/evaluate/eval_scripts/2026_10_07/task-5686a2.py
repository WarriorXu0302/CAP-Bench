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
TASK_ID = "task-5686a2"
TASK_DESCRIPTION = "I am considering purchasing a used Toyota Sienna (2021-2024 models) and would like to identify any common issues or recurring problems specific to these model years.\n\nFirst, search for Sienna reviews on Car and Driver. Focus on finding a 'Long-Term Road Test' or a comprehensive review published after 2021. Identify and record at least two significant drawbacks ('Lows') mentioned by the editors.\n\nNext, search Reddit for real-world feedback from owners concerning these two identified drawbacks. Locate the most discussed (highest comment count) relevant posts and summarize the solutions or workarounds proposed by the community.\n\nFinally, to potentially verify these issues firsthand, use Google Maps to find the official Toyota dealership closest to downtown San Francisco, CA, and calculate the estimated driving time to reach it.\n\nOutput: The drawbacks mentioned by Car and Driver along with the article link; For the highest-engagement Reddit post, provide its title, comment count, link, and a summary of the main solutions discussed; The dealership's name, address, distance from downtown San Francisco, and the estimated driving time."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CarAndDriverInfo(BaseModel):
    """Information extracted from Car and Driver review"""
    drawback_1: Optional[str] = None
    drawback_2: Optional[str] = None
    article_url: Optional[str] = None


class RedditPostInfo(BaseModel):
    """Information extracted about the highest-engagement Reddit post"""
    post_title: Optional[str] = None
    comment_count: Optional[int] = None
    post_url: Optional[str] = None
    solutions_summary: Optional[str] = None


class DealershipInfo(BaseModel):
    """Information extracted about the Toyota dealership"""
    dealership_name: Optional[str] = None
    address: Optional[str] = None
    distance_text: Optional[str] = None
    driving_time: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_caranddriver_from_answer() -> str:
    return """
Extract the Car and Driver review information for Toyota Sienna from the answer.

Return:
- drawback_1: the first drawback/low mentioned by the editors exactly as stated. If not present, set null.
- drawback_2: the second drawback/low mentioned by the editors exactly as stated. If not present, set null.
- article_url: the URL of the Car and Driver article. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_reddit_from_answer() -> str:
    return """
Extract the Reddit post information from the answer about the highest-engagement post discussing Toyota Sienna issues.

Return:
- post_title: the title of the Reddit post exactly as stated. If not present, set null.
- comment_count: the number of comments as an integer. If not present, set null.
- post_url: the URL of the Reddit post. If not present, set null.
- solutions_summary: the summary of solutions/workarounds discussed in the post. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_dealership_from_answer() -> str:
    return """
Extract the Toyota dealership information from the answer.

Return:
- dealership_name: the name of the dealership exactly as stated. If not present, set null.
- address: the address of the dealership. If not present, set null.
- distance_text: the distance from downtown San Francisco exactly as stated (include units). If not present, set null.
- driving_time: the estimated driving time exactly as stated. If not present, set null.

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


def contains_url_pattern(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    pattern = rf'https?://[^\s]*{re.escape(domain)}[^\s]*'
    return bool(re.search(pattern, text.lower()))


def extract_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\d+', str(text))
    if not m:
        return None
    try:
        return int(m.group())
    except Exception:
        return None


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    has_num = bool(re.search(r'\d', text))
    has_unit = has_any_ci(text, ['mile', 'mi', 'km', 'meter'])
    return has_num and has_unit


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    has_num = bool(re.search(r'\d', text))
    has_unit = has_any_ci(text, ['min', 'hour', 'hr', 'h'])
    return has_num and has_unit


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
    caranddriver_info = await evaluator.extract(
        prompt=prompt_extract_caranddriver_from_answer(),
        template_class=CarAndDriverInfo,
        extraction_name="caranddriver_info"
    )

    reddit_info = await evaluator.extract(
        prompt=prompt_extract_reddit_from_answer(),
        template_class=RedditPostInfo,
        extraction_name="reddit_post_info"
    )

    dealership_info = await evaluator.extract(
        prompt=prompt_extract_dealership_from_answer(),
        template_class=DealershipInfo,
        extraction_name="dealership_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Car and Driver section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver review for Toyota Sienna (2021-2024)",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F1:A3 - Navigate/search for Sienna review
    caranddriver_search_ok = (has_any_ci(answer, ['car and driver']) and
                              has_any_ci(answer, ['sienna']))
    evaluator.add_custom_node(
        result=bool(caranddriver_search_ok),
        id="caranddriver_action_search",
        desc="[Action Node] caranddriver.com:F1:A3 - Search for and navigate to Toyota Sienna review on Car and Driver",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F1:P2 - Extract at least 2 drawbacks from 'Lows'
    has_drawback_1 = bool(caranddriver_info and caranddriver_info.drawback_1 and caranddriver_info.drawback_1.strip())
    has_drawback_2 = bool(caranddriver_info and caranddriver_info.drawback_2 and caranddriver_info.drawback_2.strip())
    lows_mentioned = has_any_ci(answer, ['lows', 'drawbacks', 'cons', 'disadvantages'])

    drawbacks_ok = has_drawback_1 and has_drawback_2 and lows_mentioned
    evaluator.add_custom_node(
        result=bool(drawbacks_ok),
        id="caranddriver_perception_lows",
        desc="[Perception Node] caranddriver.com:F1:P2 - Identify at least 2 significant drawbacks from the 'Lows' section",
        parent=caranddriver_node,
        critical=False
    )

    # Check for article URL
    has_url = bool(caranddriver_info and caranddriver_info.article_url and
                   contains_url_pattern(caranddriver_info.article_url, 'caranddriver'))
    evaluator.add_custom_node(
        result=bool(has_url),
        id="caranddriver_article_url",
        desc="Provides the Car and Driver article URL",
        parent=caranddriver_node,
        critical=False
    )

    # Check for long-term or comprehensive review mention
    review_type_ok = has_any_ci(answer, ['long-term', 'long term', 'comprehensive', 'road test'])
    evaluator.add_custom_node(
        result=bool(review_type_ok),
        id="caranddriver_review_type",
        desc="Mentions finding a Long-Term Road Test or comprehensive review",
        parent=caranddriver_node,
        critical=False
    )

    # 3.2 Reddit section
    reddit_node = evaluator.add_sequential(
        id="reddit_section",
        desc="Reddit community feedback on Toyota Sienna issues",
        parent=root,
        critical=False
    )

    # [Action Node] reddit.com:F1:A1 - Search Reddit for the identified drawbacks
    reddit_search_ok = (has_any_ci(answer, ['reddit']) and
                        (has_any_ci(answer, ['search']) or has_any_ci(answer, ['found', 'post'])))

    # Check if the Reddit search relates to Car and Driver drawbacks
    drawback_keywords_in_reddit = False
    if caranddriver_info and (caranddriver_info.drawback_1 or caranddriver_info.drawback_2):
        drawback_words = []
        if caranddriver_info.drawback_1:
            drawback_words.extend(caranddriver_info.drawback_1.lower().split())
        if caranddriver_info.drawback_2:
            drawback_words.extend(caranddriver_info.drawback_2.lower().split())

        # Check if any significant keyword from drawbacks appears in Reddit context
        for word in drawback_words:
            if len(word) > 4 and has_any_ci(answer, [word]):
                drawback_keywords_in_reddit = True
                break

    evaluator.add_custom_node(
        result=bool(reddit_search_ok),
        id="reddit_action_search",
        desc="[Action Node] reddit.com:F1:A1 - Search Reddit for real-world feedback on the identified drawbacks",
        parent=reddit_node,
        critical=False
    )

    # [Perception Node] reddit.com:F1:P2 - Identify highest comment count post
    has_comment_count = bool(reddit_info and reddit_info.comment_count is not None)
    comment_count_mentioned = has_any_ci(answer, ['comment', 'discussion', 'engagement'])

    evaluator.add_custom_node(
        result=bool(has_comment_count and comment_count_mentioned),
        id="reddit_perception_top_post",
        desc="[Perception Node] reddit.com:F1:P2 - Locate the most discussed post (highest comment count)",
        parent=reddit_node,
        critical=False
    )

    # [Perception Node] reddit.com:F2:P1 - Summarize solutions from comments
    has_solutions = bool(reddit_info and reddit_info.solutions_summary and reddit_info.solutions_summary.strip())
    solutions_keywords = has_any_ci(answer, ['solution', 'fix', 'workaround', 'resolve', 'address'])

    evaluator.add_custom_node(
        result=bool(has_solutions and solutions_keywords),
        id="reddit_perception_solutions",
        desc="[Perception Node] reddit.com:F2:P1 - Summarize solutions or workarounds discussed by the community",
        parent=reddit_node,
        critical=False
    )

    # Check for Reddit post details
    has_post_title = bool(reddit_info and reddit_info.post_title and reddit_info.post_title.strip())
    has_post_url = bool(reddit_info and reddit_info.post_url and contains_url_pattern(reddit_info.post_url, 'reddit'))

    evaluator.add_custom_node(
        result=bool(has_post_title and has_post_url),
        id="reddit_post_details",
        desc="Provides Reddit post title and URL",
        parent=reddit_node,
        critical=False
    )

    # 3.3 Google Maps section
    gmaps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps search for Toyota dealership near downtown San Francisco",
        parent=root,
        critical=False
    )

    # [Action Node] google.com/maps:F1:A2 - Search for Toyota dealership
    gmaps_search_ok = (has_any_ci(answer, ['google maps', 'maps']) and
                       has_any_ci(answer, ['toyota', 'dealership']) and
                       has_any_ci(answer, ['san francisco', 'sf']))

    evaluator.add_custom_node(
        result=bool(gmaps_search_ok),
        id="gmaps_action_search",
        desc="[Action Node] google.com/maps:F1:A2 - Search for official Toyota dealership closest to downtown San Francisco",
        parent=gmaps_node,
        critical=False
    )

    # [Action Node] google.com/maps:F2:A5 - Calculate driving route
    has_driving_time = bool(dealership_info and dealership_info.driving_time and
                           looks_like_time(dealership_info.driving_time))
    route_keywords = has_any_ci(answer, ['driving time', 'drive', 'route', 'directions'])

    evaluator.add_custom_node(
        result=bool(has_driving_time and route_keywords),
        id="gmaps_action_route",
        desc="[Action Node] google.com/maps:F2:A5 - Calculate estimated driving time from downtown San Francisco",
        parent=gmaps_node,
        critical=False
    )

    # Check for dealership details
    has_name = bool(dealership_info and dealership_info.dealership_name and
                    dealership_info.dealership_name.strip())
    has_address = bool(dealership_info and dealership_info.address and
                      dealership_info.address.strip())
    has_distance = bool(dealership_info and dealership_info.distance_text and
                       looks_like_distance(dealership_info.distance_text))

    evaluator.add_custom_node(
        result=bool(has_name and has_address),
        id="gmaps_dealership_details",
        desc="Provides dealership name and address",
        parent=gmaps_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_distance),
        id="gmaps_distance",
        desc="Provides distance from downtown San Francisco",
        parent=gmaps_node,
        critical=False
    )

    # Check that it's an official Toyota dealership
    official_mention = has_any_ci(answer, ['official', 'authorized', 'toyota'])
    evaluator.add_custom_node(
        result=bool(official_mention and has_name),
        id="gmaps_official_dealership",
        desc="Confirms the dealership is an official Toyota dealership",
        parent=gmaps_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
