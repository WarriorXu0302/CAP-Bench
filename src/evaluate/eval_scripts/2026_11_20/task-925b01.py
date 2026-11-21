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
TASK_ID = "task-925b01"
TASK_DESCRIPTION = 'I’m a food blogger, and for my next family gathering I want to recreate the most authentic Bolognese ragù. There are too many “modernized” versions online, so I want to trace it back to the source.\n\nFirst, go to Wikipedia and look up the **“Bolognese sauce”** entry. Review its history and/or any recipe description related to the **Accademia Italiana della Cucina**, and identify two key rules about liquid ingredients:\n1. Whether **milk** is required  \n2. Whether the traditional wine is **red or white**\n\nUsing these two non-negotiable criteria, search **“Bolognese”** on Allrecipes. Try to find a recipe with a rating above **4.6**, more than **3,000 reviews**, and an ingredient list that strictly matches the two Wikipedia liquid-ingredient requirements.  \nIf no recipe meets all conditions at once, keep “ingredient list matches both Wikipedia liquid rules” as the hard requirement, then choose the one with the highest possible rating and review count, and explicitly note how far it falls short of the **4.6 / 3000** threshold.\n\nFinally, go to YouTube and find a matching hands-on cooking video, with **over 500,000 views** and a duration of **more than 5 minutes** to ensure sufficient detail.\n\nOutput required:  \n- Wikipedia-confirmed milk requirement and wine color  \n- Allrecipes recipe title, rating, and review count  \n- YouTube video title, view count, and duration  \n- Links to each corresponding detail page'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WikipediaLiquidIngredients(BaseModel):
    """Liquid ingredient rules extracted from Wikipedia Bolognese sauce entry"""
    milk_required: Optional[str] = None
    wine_color: Optional[str] = None
    wikipedia_link: Optional[str] = None


class AllrecipesRecipe(BaseModel):
    """Recipe details extracted from Allrecipes"""
    recipe_title: Optional[str] = None
    rating: Optional[str] = None
    review_count: Optional[str] = None
    allrecipes_link: Optional[str] = None
    milk_in_ingredients: Optional[str] = None
    wine_color_in_ingredients: Optional[str] = None


class YouTubeVideo(BaseModel):
    """Video details extracted from YouTube"""
    video_title: Optional[str] = None
    view_count: Optional[str] = None
    duration: Optional[str] = None
    youtube_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_wikipedia_from_answer() -> str:
    return """
Extract the Wikipedia Bolognese sauce information from the answer.

Return:
- milk_required: whether milk is required according to Wikipedia/Accademia Italiana della Cucina (e.g., "yes", "required", "milk is used", etc.). If not mentioned, set null.
- wine_color: the traditional wine color mentioned (e.g., "white", "red", "white wine", "red wine"). If not mentioned, set null.
- wikipedia_link: the Wikipedia link provided in the answer. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_allrecipes_from_answer() -> str:
    return """
Extract the Allrecipes recipe information from the answer.

Return:
- recipe_title: the exact title of the recipe found on Allrecipes.
- rating: the rating value exactly as stated (e.g., "4.7", "4.8 stars").
- review_count: the review count exactly as stated (e.g., "3500", "3,500 reviews").
- allrecipes_link: the Allrecipes recipe link provided.
- milk_in_ingredients: whether milk is mentioned in the recipe ingredients (e.g., "yes", "contains milk", "milk included").
- wine_color_in_ingredients: the wine color in the recipe ingredients (e.g., "white", "red", "white wine", "red wine").

If any field is missing, set it to null.
"""


def prompt_extract_youtube_from_answer() -> str:
    return """
Extract the YouTube video information from the answer.

Return:
- video_title: the exact title of the YouTube video.
- view_count: the view count exactly as stated (e.g., "1.2M", "1,200,000 views", "500K").
- duration: the video duration exactly as stated (e.g., "8:45", "10 minutes").
- youtube_link: the YouTube video link provided.

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
    m = re.findall(r'(\d+(?:\.\d+)?)', text.replace(',', ''))
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def extract_view_count_number(text: Optional[str]) -> Optional[float]:
    """Extract view count handling M/K suffixes"""
    if not text:
        return None
    text_clean = text.replace(',', '').lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*([mk])?', text_clean)
    if not m:
        return None
    try:
        num = float(m.group(1))
        suffix = m.group(2)
        if suffix == 'k':
            return num * 1000
        elif suffix == 'm':
            return num * 1000000
        return num
    except Exception:
        return None


def parse_duration_to_minutes(text: Optional[str]) -> Optional[float]:
    """Parse duration like '8:45', '10 minutes', '5:30' to total minutes"""
    if not text:
        return None
    # Try MM:SS format
    m = re.search(r'(\d+):(\d+)', text)
    if m:
        try:
            minutes = int(m.group(1))
            seconds = int(m.group(2))
            return minutes + seconds / 60.0
        except Exception:
            pass
    # Try just minutes
    m = re.search(r'(\d+(?:\.\d+)?)\s*min', text.lower())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None


def looks_like_valid_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return ci_contains(text, domain) and ('http://' in text.lower() or 'https://' in text.lower())


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
    wiki_info = await evaluator.extract(
        prompt=prompt_extract_wikipedia_from_answer(),
        template_class=WikipediaLiquidIngredients,
        extraction_name="wikipedia_liquid_ingredients"
    )

    allrecipes_info = await evaluator.extract(
        prompt=prompt_extract_allrecipes_from_answer(),
        template_class=AllrecipesRecipe,
        extraction_name="allrecipes_recipe"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_from_answer(),
        template_class=YouTubeVideo,
        extraction_name="youtube_video"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Wikipedia section
    wikipedia_node = evaluator.add_sequential(
        id="wikipedia_section",
        desc="Wikipedia Bolognese sauce entry - Accademia Italiana della Cucina liquid ingredient rules",
        parent=root,
        critical=False
    )

    # [Action Node] wikipedia.org:F1:A20 - Navigate to Bolognese sauce entry
    wiki_link_ok = looks_like_valid_url(wiki_info.wikipedia_link, 'wikipedia.org')
    wiki_mention_ok = has_any_ci(answer, ['wikipedia', 'wiki'])
    bolognese_mention_ok = has_any_ci(answer, ['bolognese'])

    evaluator.add_custom_node(
        result=bool(wiki_link_ok or (wiki_mention_ok and bolognese_mention_ok)),
        id="wikipedia_action_navigate",
        desc="[Action Node] wikipedia.org:F1:A20 - Navigate to Wikipedia Bolognese sauce entry",
        parent=wikipedia_node,
        critical=False
    )

    # [Action Node] wikipedia.org:F2:A5 - Tab switching / section navigation
    section_navigation_ok = has_any_ci(answer, ['accademia italiana della cucina', 'history', 'recipe', 'registered'])
    evaluator.add_custom_node(
        result=bool(section_navigation_ok),
        id="wikipedia_action_tab_switch",
        desc="[Action Node] wikipedia.org:F2:A5 - Navigate through sections/tabs to find Accademia recipe information",
        parent=wikipedia_node,
        critical=False
    )

    # [Perception Node] wikipedia.org:F2:P9 - Extract milk and wine color rules
    milk_extracted_ok = bool(wiki_info.milk_required and wiki_info.milk_required.strip())
    wine_extracted_ok = bool(wiki_info.wine_color and wiki_info.wine_color.strip())

    milk_rule_ok = milk_extracted_ok and has_any_ci(wiki_info.milk_required, ['yes', 'required', 'milk'])
    wine_color_ok = wine_extracted_ok and has_any_ci(wiki_info.wine_color, ['white', 'red'])

    evaluator.add_custom_node(
        result=bool(milk_rule_ok and wine_color_ok),
        id="wikipedia_perception_liquid_rules",
        desc="[Perception Node] wikipedia.org:F2:P9 - Extract milk requirement and traditional wine color from recipe description",
        parent=wikipedia_node,
        critical=False
    )

    # 3.2 Allrecipes section
    allrecipes_node = evaluator.add_sequential(
        id="allrecipes_section",
        desc="Allrecipes search for Bolognese recipe matching Wikipedia liquid ingredient criteria",
        parent=root,
        critical=False
    )

    # [Action Node] allrecipes.com:F1:A2 - Search for Bolognese
    allrecipes_mention_ok = has_any_ci(answer, ['allrecipes', 'all recipes'])
    search_performed_ok = allrecipes_mention_ok and bolognese_mention_ok

    evaluator.add_custom_node(
        result=bool(search_performed_ok),
        id="allrecipes_action_search",
        desc="[Action Node] allrecipes.com:F1:A2 - Search for 'Bolognese' on Allrecipes",
        parent=allrecipes_node,
        critical=False
    )

    # [Action Node] allrecipes.com:F3:A3 - Click recipe card to view details
    recipe_link_ok = looks_like_valid_url(allrecipes_info.allrecipes_link, 'allrecipes.com')
    recipe_title_ok = bool(allrecipes_info.recipe_title and allrecipes_info.recipe_title.strip())

    evaluator.add_custom_node(
        result=bool(recipe_link_ok or recipe_title_ok),
        id="allrecipes_action_click_card",
        desc="[Action Node] allrecipes.com:F3:A3 - Click recipe card to view ingredient details",
        parent=allrecipes_node,
        critical=False
    )

    # Check if recipe matches Wikipedia liquid ingredient criteria
    recipe_milk_ok = bool(allrecipes_info.milk_in_ingredients and
                         has_any_ci(allrecipes_info.milk_in_ingredients, ['yes', 'milk', 'contains']))
    recipe_wine_ok = bool(allrecipes_info.wine_color_in_ingredients and
                         has_any_ci(allrecipes_info.wine_color_in_ingredients, ['white', 'red']))

    # Check if recipe matches Wikipedia rules
    ingredients_match_wiki = False
    if milk_rule_ok and wine_color_ok and recipe_milk_ok and recipe_wine_ok:
        wiki_wine = (wiki_info.wine_color or '').lower()
        recipe_wine = (allrecipes_info.wine_color_in_ingredients or '').lower()
        ingredients_match_wiki = ('white' in wiki_wine and 'white' in recipe_wine) or \
                                ('red' in wiki_wine and 'red' in recipe_wine)

    evaluator.add_custom_node(
        result=bool(ingredients_match_wiki),
        id="allrecipes_ingredients_match_wiki",
        desc="Recipe ingredients strictly match Wikipedia liquid rules (milk + correct wine color)",
        parent=allrecipes_node,
        critical=False
    )

    # Check rating and review count thresholds
    rating_num = extract_float(allrecipes_info.rating)
    review_num = extract_float(allrecipes_info.review_count)

    rating_ok = rating_num is not None and rating_num > 4.6
    reviews_ok = review_num is not None and review_num > 3000

    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="allrecipes_rating_threshold",
        desc="Recipe rating above 4.6",
        parent=allrecipes_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(reviews_ok),
        id="allrecipes_review_count_threshold",
        desc="Recipe has more than 3,000 reviews",
        parent=allrecipes_node,
        critical=False
    )

    # 3.3 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube search for Bolognese cooking video with sufficient detail",
        parent=root,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Identify traditional/authentic video from thumbnail/title
    youtube_mention_ok = has_any_ci(answer, ['youtube'])
    video_title_ok = bool(youtube_info.video_title and youtube_info.video_title.strip())
    traditional_keywords_ok = has_any_ci(youtube_info.video_title or '',
                                        ['traditional', 'authentic', 'slow', 'classic', 'italian', 'bolognese'])

    evaluator.add_custom_node(
        result=bool(traditional_keywords_ok and video_title_ok),
        id="youtube_perception_traditional_style",
        desc="[Perception Node] youtube.com:F1:P4 - Identify traditional/authentic cooking style from title or thumbnail",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P11 - Extract duration from thumbnail/card
    duration_minutes = parse_duration_to_minutes(youtube_info.duration)
    duration_ok = duration_minutes is not None and duration_minutes > 5.0

    evaluator.add_custom_node(
        result=bool(duration_ok),
        id="youtube_perception_duration_ocr",
        desc="[Perception Node] youtube.com:F1:P11 - Extract video duration (must be over 5 minutes)",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll/filter to find high view count video
    view_count_num = extract_view_count_number(youtube_info.view_count)
    views_ok = view_count_num is not None and view_count_num > 500000

    evaluator.add_custom_node(
        result=bool(views_ok),
        id="youtube_action_scroll_filter",
        desc="[Action Node] youtube.com:F1:A69 - Scroll/filter video list to find video with over 500,000 views",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F2:A9 - Quality check (optional, represented by link validation)
    youtube_link_ok = looks_like_valid_url(youtube_info.youtube_link, 'youtube.com')

    evaluator.add_custom_node(
        result=bool(youtube_link_ok),
        id="youtube_action_quality_check",
        desc="[Action Node] youtube.com:F2:A9 - Verify video quality/playback settings",
        parent=youtube_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
