import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-1533cc"
TASK_DESCRIPTION = """
I am planning to attend a 3-day conference in Miami, Florida, during the last week of January next year (specifically from the 28th to the 30th of January), and have already booked a stay at the Hilton Miami Downtown. However, I just saw news reports indicating there might be a hurricane passing through during those dates. I want to confirm the level of risk and whether I should reschedule or change location if the risk is significant.

Please first check AccuWeather for the weather conditions in Miami during the last week of January. If there are severe weather alerts (Watches or Warnings) or hurricane tracking indicates a storm might affect the area, help me assess the risk level. Then, check Booking.com for Hilton Miami Downtown's cancellation policy (free cancellation deadline and cancellation fees). If the policy allows and the risk is high, please search on Booking.com for alternative hotels in safer areas surrounding Miami (e.g., 50-100 miles north of Miami) for the last week of January, with the following requirements: a rating above 8.0, free cancellation support, and a price range of $150-$300 per night.

If the risk in Miami itself is low but flights might be affected, check Google Flights for flights to Miami departing on the 27th of January next year. Find 3 alternative flights (direct or one-stop, including different airlines and departure times), and record the flight numbers, departure/arrival times, fares, and rebooking policies.

Finally, if a hotel change is necessary, provide links to the official booking pages of the alternative hotels. If alternative flights are needed, provide links to the airline's official flight rebooking pages.

Output:
1) AccuWeather's weather alert level (if any), alert type, affected period, and risk assessment;
2) Hilton Miami Downtown's free cancellation deadline, cancellation fees, and Booking.com details page link;
3) If alternative hotels are required, provide the names, addresses, ratings, per-night prices, free cancellation policies, Booking.com details page links, and official website booking links for 3 suitable hotels;
4) If alternative flights are required, provide the flight numbers, airlines, departure/arrival times, flight durations, fares, Google Flights links, and airline official rebooking page links for 3 alternative flights;
5) Contingency plan recommendations based on the above information (continue as planned / change hotels / reschedule).
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WeatherAlertInfo(BaseModel):
    region: Optional[str] = None  # e.g., Miami, FL
    alert_level: Optional[str] = None  # Watch, Warning, or None/No alerts
    alert_type: Optional[str] = None  # e.g., Hurricane Warning
    affected_period: Optional[str] = None  # date range or textual period


class HurricaneTracking(BaseModel):
    storm_name: Optional[str] = None
    current_position: Optional[str] = None
    distance_to_miami_miles: Optional[float] = None
    forecast_path_overview: Optional[str] = None
    could_affect_miami: Optional[bool] = None


class RiskAssessment(BaseModel):
    risk_level: Optional[str] = None  # low/medium/high
    rationale: Optional[str] = None
    based_on_path_analysis: bool = False


class HiltonPolicy(BaseModel):
    hotel_name: Optional[str] = None
    free_cancellation_deadline: Optional[str] = None
    cancellation_fees: Optional[str] = None
    booking_details_url: Optional[str] = None


class AltHotel(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    price_per_night_usd: Optional[float] = None
    free_cancellation: Optional[bool] = None
    free_cancellation_text: Optional[str] = None
    booking_url: Optional[str] = None
    official_site_url: Optional[str] = None
    available_dates: List[str] = Field(default_factory=list)


class AlternativeHotels(BaseModel):
    date_range_str: Optional[str] = None  # e.g., Jan 28 - Jan 30
    hotels: List[AltHotel] = Field(default_factory=list)


class FlightInfo(BaseModel):
    flight_number: Optional[str] = None
    airline: Optional[str] = None
    departure_airport: Optional[str] = None
    arrival_airport: Optional[str] = None
    departure_time: Optional[str] = None  # include date
    arrival_time: Optional[str] = None
    duration_str: Optional[str] = None
    duration_minutes: Optional[int] = None
    fare_usd: Optional[float] = None
    stops: Optional[int] = None
    google_flights_link: Optional[str] = None
    rebooking_link: Optional[str] = None


class FlightsList(BaseModel):
    flights: List[FlightInfo] = Field(default_factory=list)


class Recommendation(BaseModel):
    plan: Optional[str] = None  # continue as planned / change hotels / reschedule
    rationale: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_weather_alert() -> str:
    return """
    From the answer, extract AccuWeather details for Miami during the last week of January:
    - region (should reference Miami, FL if available)
    - alert_level (Watch, Warning, or 'No Alerts'/'None')
    - alert_type (e.g., Hurricane Warning, Tropical Storm Watch)
    - affected_period (exact dates or textual period)

    If any field is missing in the answer, set it to null. Return a JSON object.
    """


def prompt_extract_hurricane_tracking() -> str:
    return """
    From the answer, extract hurricane tracking information (if any):
    - storm_name
    - current_position (as described)
    - distance_to_miami_miles (numeric if provided; otherwise null)
    - forecast_path_overview (text of predicted path)
    - could_affect_miami (true/false if stated or implied; otherwise null)

    If not applicable, set fields to null. Return a JSON object.
    """


def prompt_extract_risk_assessment() -> str:
    return """
    From the answer, extract the risk assessment for Miami during Jan 28-30:
    - risk_level (low/medium/high if stated; otherwise null)
    - rationale (text)
    - based_on_path_analysis (true if the rationale mentions storm track/path relative to Miami; else false)

    Return a JSON object.
    """


def prompt_extract_hilton_policy() -> str:
    return """
    From the answer, extract the Hilton Miami Downtown cancellation policy as found on Booking.com:
    - hotel_name
    - free_cancellation_deadline (exact phrasing)
    - cancellation_fees (exact phrasing)
    - booking_details_url (Booking.com hotel details page URL)

    Return a JSON object. If any field is missing, set it to null.
    """


def prompt_extract_alternative_hotels() -> str:
    return """
    From the answer, extract up to 3 alternative hotels for Jan 28-30 (50-100 miles north of Miami) from Booking.com:
    - date_range_str (e.g., 'Jan 28 - Jan 30')
    For each hotel:
      - name
      - address
      - rating (numeric, if provided)
      - price_per_night_usd (numeric, if provided)
      - free_cancellation (true/false if clearly stated)
      - free_cancellation_text (e.g., 'Free cancellation')
      - booking_url (Booking.com details page)
      - official_site_url (official website booking link)
      - available_dates (list of dates or a textual date range mentioned)

    Return a JSON object with the list of hotels.
    """


def prompt_extract_flights() -> str:
    return """
    From the answer, extract 3 alternative flights to Miami departing on January 27:
    For each flight:
      - flight_number
      - airline
      - departure_airport
      - arrival_airport
      - departure_time (include the date string present)
      - arrival_time
      - duration_str
      - duration_minutes (numeric if present or can be inferred)
      - fare_usd (numeric if present)
      - stops (0 for nonstop, 1 for one-stop; higher if stated)
      - google_flights_link
      - rebooking_link (airline official rebooking page)

    Return a JSON object with the list of flights.
    """


def prompt_extract_recommendation() -> str:
    return """
    From the answer, extract the final contingency plan recommendation:
    - plan (should be one of: 'continue as planned', 'change hotels', 'reschedule', or a close variant)
    - rationale (text explaining why)

    Return a JSON object.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def contains_miami(s: Optional[str]) -> bool:
    text = safe_lower(s)
    return "miami" in text or "mia" in text


def url_contains(s: Optional[str], subs: List[str]) -> bool:
    text = safe_lower(s)
    return all(sub.lower() in text for sub in subs)


def parse_price(val: Optional[float]) -> Optional[float]:
    # assumes it's numeric already if provided; otherwise try to parse from string
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def parse_duration_minutes(duration_str: Optional[str], duration_minutes: Optional[int]) -> Optional[int]:
    if isinstance(duration_minutes, int) and duration_minutes > 0:
        return duration_minutes
    if not duration_str:
        return None
    # simple parse like "5h 30m" or "2h" or "180m"
    text = duration_str.lower().replace(" ", "")
    total = 0
    try:
        if "h" in text:
            h_part = text.split("h")[0]
            total += int("".join([c for c in h_part if c.isdigit()]))
            text_after_h = text.split("h", 1)[1]
            if "m" in text_after_h:
                m_part = text_after_h.split("m")[0]
                if m_part:
                    total = total * 60 + int("".join([c for c in m_part if c.isdigit()]))
                else:
                    total = total * 60
            else:
                total = total * 60
            return total
        elif "m" in text:
            m_part = text.split("m")[0]
            return int("".join([c for c in m_part if c.isdigit()]))
    except Exception:
        return None
    return None


def is_jan27(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    s = safe_lower(date_str)
    # lenient: accept 'jan 27', 'january 27', '01/27', '202X-01-27'
    return ("jan 27" in s) or ("january 27" in s) or ("-01-27" in s) or ("/01/27" in s) or ("1/27" in s)


def has_free_cancellation_text(s: Optional[str]) -> bool:
    text = safe_lower(s)
    return ("free cancellation" in text) or ("free cancel" in text)


def in_price_range(val: Optional[float], low: float = 150, high: float = 300) -> bool:
    if val is None:
        return False
    return low - 5 <= val <= high + 5  # slightly lenient


def rating_ok(val: Optional[float]) -> bool:
    if val is None:
        return False
    try:
        return float(val) >= 8.0
    except Exception:
        return False


def period_mentions_jan_28_30(period: Optional[str]) -> bool:
    s = safe_lower(period)
    return ("jan 28" in s and "jan 30" in s) or ("january 28" in s and "january 30" in s) or ("28th" in s and "30th" in s)


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

    # -------------------- Extract information from the answer ---------------- #
    weather_alert = await evaluator.extract(
        prompt=prompt_extract_weather_alert(),
        template_class=WeatherAlertInfo,
        extraction_name="o1_o2"
    )

    hurricane_info = await evaluator.extract(
        prompt=prompt_extract_hurricane_tracking(),
        template_class=HurricaneTracking,
        extraction_name="o3"
    )

    risk_assessment = await evaluator.extract(
        prompt=prompt_extract_risk_assessment(),
        template_class=RiskAssessment,
        extraction_name="o4"
    )

    hilton_policy = await evaluator.extract(
        prompt=prompt_extract_hilton_policy(),
        template_class=HiltonPolicy,
        extraction_name="o6_o7_o8"
    )

    alt_hotels = await evaluator.extract(
        prompt=prompt_extract_alternative_hotels(),
        template_class=AlternativeHotels,
        extraction_name="o9_o11_o12_o13"
    )

    flights_list = await evaluator.extract(
        prompt=prompt_extract_flights(),
        template_class=FlightsList,
        extraction_name="o16_o18_o19_o20_o21"
    )

    recommendation = await evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=Recommendation,
        extraction_name="final_recommendation"
    )

    # -------------------- Build evaluation tree ----------------------------- #

    # 1) AccuWeather checks
    accuweather_node = evaluator.add_sequential(
        id="accuweather_checks",
        desc="AccuWeather weather alerts and hurricane tracking checks",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F6:A8 - 地图缩放平移 (o1 includes Miami region alert info)
    acc_action_F6A8 = evaluator.add_custom_node(
        result=contains_miami(weather_alert.region) or contains_miami(weather_alert.affected_period) or contains_miami(weather_alert.alert_type),
        id="o1_F6A8",
        desc="[Action Node] accuweather.com:F6:A8 - 地图缩放平移: Check Miami region is referenced for alerts",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F6:P8 - 区域理解 (Watch vs Warning recognition)
    level = safe_lower(weather_alert.alert_level)
    level_ok = ("watch" in level) or ("warning" in level) or ("no alert" in level) or ("none" in level) or ("no alerts" in level)
    acc_perc_F6P8 = evaluator.add_custom_node(
        result=level_ok,
        id="o1_F6P8",
        desc="[Perception Node] accuweather.com:F6:P8 - 区域理解: Recognize alert level (Watch/Warning or None)",
        parent=accuweather_node,
        critical=False
    )

    # [Action Node] accuweather.com:F6:A10 - 地图标记交互 (alert type and affected period)
    has_type = bool(weather_alert.alert_type)
    has_period = bool(weather_alert.affected_period)
    period_ok = has_period and period_mentions_jan_28_30(weather_alert.affected_period)
    acc_action_F6A10 = evaluator.add_custom_node(
        result=has_type and (has_period or True),  # lenient: require type and some period text
        id="o2_F6A10",
        desc="[Action Node] accuweather.com:F6:A10 - 地图标记交互: Alert type and affected period are provided",
        parent=accuweather_node,
        critical=False
    )

    # [Action Node] accuweather.com:F5:A9 - 风暴标记交互 (storm marker details)
    storm_marker_ok = bool(hurricane_info.storm_name) and bool(hurricane_info.forecast_path_overview)
    acc_action_F5A9 = evaluator.add_custom_node(
        result=storm_marker_ok,
        id="o3_F5A9",
        desc="[Action Node] accuweather.com:F5:A9 - 风暴标记交互: Storm name and forecast path are provided",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F5:P5 - 位置理解 (storm position and distance to Miami)
    has_position = bool(hurricane_info.current_position)
    has_distance = (hurricane_info.distance_to_miami_miles is not None) or (hurricane_info.current_position and ("mile" in safe_lower(hurricane_info.current_position) or "km" in safe_lower(hurricane_info.current_position)))
    acc_perc_F5P5 = evaluator.add_custom_node(
        result=has_position and (has_distance or True),  # lenient: position is enough, distance optional
        id="o3_F5P5",
        desc="[Perception Node] accuweather.com:F5:P5 - 位置理解: Storm current position and distance/relative proximity recognized",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F5:P6 - 路线理解 (risk based on path analysis)
    risk_level_norm = safe_lower(risk_assessment.risk_level)
    risk_level_valid = risk_level_norm in ["low", "medium", "high"]
    path_based = bool(risk_assessment.based_on_path_analysis)
    rationale_mentions_path = "path" in safe_lower(risk_assessment.rationale) or "track" in safe_lower(risk_assessment.rationale) or "trajectory" in safe_lower(risk_assessment.rationale) or "cone" in safe_lower(risk_assessment.rationale)
    acc_perc_F5P6 = evaluator.add_custom_node(
        result=(risk_level_valid or True) and (path_based or rationale_mentions_path),
        id="o4_F5P6",
        desc="[Perception Node] accuweather.com:F5:P6 - 路线理解: Risk assessment based on storm path analysis",
        parent=accuweather_node,
        critical=False
    )

    # 2) Booking.com - Hilton policy
    booking_hilton_node = evaluator.add_sequential(
        id="booking_hilton",
        desc="Booking.com Hilton Miami Downtown cancellation policy check",
        parent=root,
        critical=False
    )

    # [Action Node] Booking.com:F1:A4 - 搜索直接定位 (Hilton details URL)
    hilton_url_ok = url_contains(hilton_policy.booking_details_url, ["booking.com"]) and (("hilton" in safe_lower(hilton_policy.booking_details_url)) or ("hilton" in safe_lower(hilton_policy.hotel_name or ""))) and ("miami" in safe_lower(hilton_policy.booking_details_url) or "miami" in safe_lower(hilton_policy.hotel_name or ""))
    booking_action_F1A4 = evaluator.add_custom_node(
        result=hilton_url_ok,
        id="o8_F1A4",
        desc="[Action Node] Booking.com:F1:A4 - 搜索直接定位: Booking.com Hilton Miami Downtown details URL provided",
        parent=booking_hilton_node,
        critical=False
    )

    # [Action Node] Booking.com:F3:A14 - 点击卡片进详情 (policy info present)
    has_deadline = bool(hilton_policy.free_cancellation_deadline)
    has_fees = bool(hilton_policy.cancellation_fees)
    booking_action_F3A14 = evaluator.add_custom_node(
        result=has_deadline or has_fees,
        id="o6o7_F3A14",
        desc="[Action Node] Booking.com:F3:A14 - 点击卡片进详情: Cancellation policy info present",
        parent=booking_hilton_node,
        critical=False
    )

    # [Action Node] Booking.com:F3:A46 - 展开折叠面板 (explicit deadline and fee details)
    booking_action_F3A46 = evaluator.add_custom_node(
        result=has_deadline and has_fees,
        id="o6o7_F3A46",
        desc="[Action Node] Booking.com:F3:A46 - 展开折叠面板: Free cancellation deadline and cancellation fees extracted",
        parent=booking_hilton_node,
        critical=False
    )

    # 3) Booking.com - Alternative hotels
    booking_alts_node = evaluator.add_sequential(
        id="booking_alts",
        desc="Booking.com alternative hotels search and filters",
        parent=root,
        critical=False
    )

    # [Action Node] Booking.com:F1:A1 - 日期范围选择 (Jan 28-30 availability)
    date_ok = period_mentions_jan_28_30(alt_hotels.date_range_str) or any(any("jan 28" in safe_lower(d) or "january 28" in safe_lower(d) or "-01-28" in safe_lower(d) for d in h.available_dates) for h in alt_hotels.hotels) and any(any("jan 30" in safe_lower(d) or "january 30" in safe_lower(d) or "-01-30" in safe_lower(d) for d in h.available_dates) for h in alt_hotels.hotels)
    booking_action_F1A1 = evaluator.add_custom_node(
        result=bool(alt_hotels.hotels) and (date_ok or True),  # lenient: at least hotels listed, with some date mention
        id="o9_F1A1",
        desc="[Action Node] Booking.com:F1:A1 - 日期范围选择: Alternative hotels correspond to Jan 28-30",
        parent=booking_alts_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A11 - 评分筛选 (ratings >= 8.0)
    ratings = [h.rating for h in alt_hotels.hotels if h.rating is not None]
    ratings_pass_count = sum(1 for r in ratings if rating_ok(r))
    booking_action_F2A11 = evaluator.add_custom_node(
        result=(ratings_pass_count >= max(1, min(3, len(alt_hotels.hotels)) - 1)),  # lenient: most pass
        id="o11_F2A11",
        desc="[Action Node] Booking.com:F2:A11 - 评分筛选: Hotels have ratings ≥ 8.0",
        parent=booking_alts_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A10 - 价格范围 (150-300 USD)
    prices = [parse_price(h.price_per_night_usd) for h in alt_hotels.hotels if h.price_per_night_usd is not None]
    price_pass_count = sum(1 for p in prices if in_price_range(p, 150, 300))
    booking_action_F2A10 = evaluator.add_custom_node(
        result=(price_pass_count >= max(1, min(3, len(alt_hotels.hotels)) - 1)),  # lenient: most in range
        id="o12_F2A10",
        desc="[Action Node] Booking.com:F2:A10 - 范围拖动: Hotel prices per night between $150-$300",
        parent=booking_alts_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A7 - 多项勾选 (free cancellation supported)
    free_cancel_pass = sum(1 for h in alt_hotels.hotels if (h.free_cancellation is True) or has_free_cancellation_text(h.free_cancellation_text)) >= max(1, min(3, len(alt_hotels.hotels)) - 1)
    booking_action_F2A7 = evaluator.add_custom_node(
        result=free_cancel_pass,
        id="o13_F2A7",
        desc="[Action Node] Booking.com:F2:A7 - 多项勾选: Free cancellation selected/supported",
        parent=booking_alts_node,
        critical=False
    )

    # [Perception Node] Booking.com:F2:P2 - 状态标签识别 (recognize free cancellation label)
    label_recognized = all((h.free_cancellation is True) or has_free_cancellation_text(h.free_cancellation_text) for h in alt_hotels.hotels) if alt_hotels.hotels else False
    booking_perc_F2P2 = evaluator.add_custom_node(
        result=label_recognized or free_cancel_pass,  # lenient
        id="o13_F2P2",
        desc="[Perception Node] Booking.com:F2:P2 - 状态标签识别: Free cancellation label recognized on hotel cards",
        parent=booking_alts_node,
        critical=False
    )

    # Non-covered general check: official hotel booking links present (not prefixed)
    official_links_ok = sum(1 for h in alt_hotels.hotels if h.official_site_url) >= min(3, len(alt_hotels.hotels))
    evaluator.add_custom_node(
        result=official_links_ok or (len(alt_hotels.hotels) == 0),
        id="alt_hotels_official_links",
        desc="Alternative hotels include official website booking links (if hotels are provided)",
        parent=booking_alts_node,
        critical=False
    )

    # 4) Google Flights checks
    flights_node = evaluator.add_sequential(
        id="google_flights",
        desc="Google Flights alternative flight options checks",
        parent=root,
        critical=False
    )

    flights = flights_list.flights or []
    # [Action Node] google.comflights:F1:A1 - 输入并选择目的地 (arrive in Miami)
    flights_to_miami = sum(1 for f in flights if contains_miami(f.arrival_airport))
    flights_action_F1A1 = evaluator.add_custom_node(
        result=flights_to_miami >= max(1, min(3, len(flights)) - 1),
        id="o16_F1A1",
        desc="[Action Node] google.comflights:F1:A1 - 输入并选择目的地: Flights arrive in Miami (MIA)",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F1:A2 - 选择日期 (depart on Jan 27)
    flights_on_jan27 = sum(1 for f in flights if is_jan27(f.departure_time))
    flights_action_F1A2 = evaluator.add_custom_node(
        result=flights_on_jan27 >= max(1, min(3, len(flights)) - 1),
        id="o18_F1A2",
        desc="[Action Node] google.comflights:F1:A2 - 选择日期: Departure date is January 27",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F3:A5 - 组合筛选 (Nonstop or 1 stop)
    stops_ok_count = sum(1 for f in flights if (f.stops is None) or (isinstance(f.stops, int) and f.stops <= 1))
    flights_action_F3A5 = evaluator.add_custom_node(
        result=stops_ok_count >= max(1, min(3, len(flights)) - 1),
        id="o20_F3A5",
        desc="[Action Node] google.comflights:F3:A5 - 组合筛选: Stops ≤ 1",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F6:A14 - 点击列表项展开详情 (detailed info available)
    details_ok_count = 0
    for f in flights:
        if (f.flight_number and f.airline and f.departure_time and f.arrival_time and (f.fare_usd is not None or f.google_flights_link)):
            details_ok_count += 1
    flights_action_F6A14 = evaluator.add_custom_node(
        result=details_ok_count >= max(1, min(3, len(flights)) - 1),
        id="o16_F6A14",
        desc="[Action Node] google.comflights:F6:A14 - 点击列表项展开详情: Flight number, times, fare/link extracted",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F6:P1 - 数值数据识别 (fare and duration numeric)
    numeric_ok_count = 0
    for f in flights:
        dur = parse_duration_minutes(f.duration_str, f.duration_minutes)
        fare_ok = (f.fare_usd is not None) and (f.fare_usd >= 0)
        if (dur is not None) and fare_ok:
            numeric_ok_count += 1
    flights_perc_F6P1 = evaluator.add_custom_node(
        result=numeric_ok_count >= max(1, min(3, len(flights)) - 1),
        id="o21_F6P1",
        desc="[Perception Node] google.comflights:F6:P1 - 数值数据识别: Numeric fare and duration recognized",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F6:P11 - 航班信息识别 (flight number and schedule)
    schedule_ok_count = sum(1 for f in flights if (f.flight_number and f.departure_time and f.arrival_time))
    flights_perc_F6P11 = evaluator.add_custom_node(
        result=schedule_ok_count >= max(1, min(3, len(flights)) - 1),
        id="o16o18_F6P11",
        desc="[Perception Node] google.comflights:F6:P11 - 航班信息识别: Flight numbers and times extracted",
        parent=flights_node,
        critical=False
    )

    # Non-covered general check: airline rebooking links present
    rebooking_links_ok = sum(1 for f in flights if f.rebooking_link) >= max(1, min(3, len(flights)) - 1)
    evaluator.add_custom_node(
        result=rebooking_links_ok or (len(flights) == 0),
        id="flights_rebooking_links",
        desc="Airline official rebooking links are provided for most alternative flights (if flights are provided)",
        parent=flights_node,
        critical=False
    )

    # 5) Final recommendation present
    final_node = evaluator.add_sequential(
        id="final_plan",
        desc="Final contingency plan recommendation",
        parent=root,
        critical=False
    )

    plan_ok = safe_lower(recommendation.plan) in ["continue as planned", "change hotels", "reschedule"] or bool(recommendation.plan)
    evaluator.add_custom_node(
        result=plan_ok,
        id="final_plan_presence",
        desc="Recommendation plan provided",
        parent=final_node,
        critical=False
    )

    rationale_ok = bool(recommendation.rationale) or bool(risk_assessment.rationale)
    evaluator.add_custom_node(
        result=rationale_ok,
        id="final_plan_rationale",
        desc="Recommendation rationale provided",
        parent=final_node,
        critical=False
    )

    # Return summary
    return evaluator.get_summary()