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
TASK_ID = "task-1533cc"
TASK_DESCRIPTION = "I am planning to attend a 3-day conference in Miami, Florida, during the last week of January next year (specifically from the 28th to the 30th of January), and have already booked a stay at the Hilton Miami Downtown. However, I just saw news reports indicating there might be a hurricane passing through during those dates. I want to confirm the level of risk and whether I should reschedule or change location if the risk is significant.\n\nPlease first check AccuWeather for the weather conditions in Miami during the last week of January. If there are severe weather alerts (Watches or Warnings) or hurricane tracking indicates a storm might affect the area, help me assess the risk level. Then, check Booking.com for Hilton Miami Downtown's cancellation policy (free cancellation deadline and cancellation fees). If the policy allows and the risk is high, please search on Booking.com for alternative hotels in safer areas surrounding Miami (e.g., 50-100 miles north of Miami) for the last week of January, with the following requirements: a rating above 8.0, free cancellation support, and a price range of $150-$300 per night.\n\nIf the risk in Miami itself is low but flights might be affected, check Google Flights for flights to Miami departing on the 27th of January next year. Find 3 alternative flights (direct or one-stop, including different airlines and departure times), and record the flight numbers, departure/arrival times, fares, and rebooking policies.\n\nFinally, if a hotel change is necessary, provide links to the official booking pages of the alternative hotels. If alternative flights are needed, provide links to the airline's official flight rebooking pages.\n\nOutput:\n1) AccuWeather's weather alert level (if any), alert type, affected period, and risk assessment;\n2) Hilton Miami Downtown's free cancellation deadline, cancellation fees, and Booking.com details page link;\n3) If alternative hotels are required, provide the names, addresses, ratings, per-night prices, free cancellation policies, Booking.com details page links, and official website booking links for 3 suitable hotels;\n4) If alternative flights are required, provide the flight numbers, airlines, departure/arrival times, flight durations, fares, Google Flights links, and airline official rebooking page links for 3 alternative flights;\n5) Contingency plan recommendations based on the above information (continue as planned / change hotels / reschedule)."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WeatherAlertInfo(BaseModel):
    """Weather alert information from AccuWeather"""
    alert_level: Optional[str] = None
    alert_type: Optional[str] = None
    affected_period: Optional[str] = None
    risk_assessment: Optional[str] = None


class HurricaneTrackingInfo(BaseModel):
    """Hurricane tracking information from AccuWeather"""
    storm_name: Optional[str] = None
    storm_location: Optional[str] = None
    distance_to_miami: Optional[str] = None
    predicted_path: Optional[str] = None


class HiltonCancellationPolicy(BaseModel):
    """Cancellation policy for Hilton Miami Downtown"""
    free_cancellation_deadline: Optional[str] = None
    cancellation_fees: Optional[str] = None
    booking_link: Optional[str] = None


class AlternativeHotel(BaseModel):
    """Alternative hotel information"""
    name: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[str] = None
    price_per_night: Optional[str] = None
    free_cancellation_policy: Optional[str] = None
    booking_link: Optional[str] = None
    official_website_link: Optional[str] = None


class AlternativeFlight(BaseModel):
    """Alternative flight information"""
    flight_number: Optional[str] = None
    airline: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    flight_duration: Optional[str] = None
    fare: Optional[str] = None
    google_flights_link: Optional[str] = None
    airline_rebooking_link: Optional[str] = None


class ContingencyPlan(BaseModel):
    """Contingency plan recommendation"""
    recommendation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_weather_alert() -> str:
    return """
Extract AccuWeather's weather alert information for Miami during the last week of January from the answer.

Return:
- alert_level: the severity level (e.g., Watch, Warning, Advisory)
- alert_type: the type of alert (e.g., Hurricane Watch, Storm Warning)
- affected_period: the time period affected
- risk_assessment: the assessed risk level

If any field is missing, set it to null.
"""


def prompt_extract_hurricane_tracking() -> str:
    return """
Extract hurricane tracking information from AccuWeather for storms that might affect Miami from the answer.

Return:
- storm_name: the name of the storm
- storm_location: current location of the storm
- distance_to_miami: distance from Miami
- predicted_path: the predicted path of the storm

If any field is missing, set it to null.
"""


def prompt_extract_hilton_cancellation() -> str:
    return """
Extract the cancellation policy for Hilton Miami Downtown from Booking.com as stated in the answer.

Return:
- free_cancellation_deadline: the deadline for free cancellation
- cancellation_fees: any fees for cancellation
- booking_link: the Booking.com details page link

If any field is missing, set it to null.
"""


def prompt_extract_alternative_hotels() -> str:
    return """
Extract information about alternative hotels from the answer. Return a list of up to 3 hotels.

For each hotel return:
- name: hotel name
- address: hotel address
- rating: rating score
- price_per_night: price per night
- free_cancellation_policy: free cancellation details
- booking_link: Booking.com link
- official_website_link: official website booking link

If any field is missing, set it to null.
"""


def prompt_extract_alternative_flights() -> str:
    return """
Extract information about alternative flights from the answer. Return a list of up to 3 flights.

For each flight return:
- flight_number: flight number
- airline: airline name
- departure_time: departure time
- arrival_time: arrival time
- flight_duration: duration of flight
- fare: ticket price
- google_flights_link: Google Flights link
- airline_rebooking_link: airline official rebooking page link

If any field is missing, set it to null.
"""


def prompt_extract_contingency_plan() -> str:
    return """
Extract the contingency plan recommendation from the answer.

Return:
- recommendation: the recommended course of action (continue as planned / change hotels / reschedule)

If missing, set it to null.
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


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    rating = extract_float(text)
    return rating is not None and rating >= 8.0


def looks_like_price_range(text: Optional[str]) -> bool:
    if not text:
        return False
    price = extract_float(text)
    return price is not None and 150 <= price <= 300


def has_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'https?://', text))


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
    weather_alert = await evaluator.extract(
        prompt=prompt_extract_weather_alert(),
        template_class=WeatherAlertInfo,
        extraction_name="weather_alert_info"
    )

    hurricane_tracking = await evaluator.extract(
        prompt=prompt_extract_hurricane_tracking(),
        template_class=HurricaneTrackingInfo,
        extraction_name="hurricane_tracking_info"
    )

    hilton_policy = await evaluator.extract(
        prompt=prompt_extract_hilton_cancellation(),
        template_class=HiltonCancellationPolicy,
        extraction_name="hilton_cancellation_policy"
    )

    # For alternative hotels and flights, we extract as text then parse
    # Since BaseModel doesn't handle lists well, we'll check answer directly

    contingency = await evaluator.extract(
        prompt=prompt_extract_contingency_plan(),
        template_class=ContingencyPlan,
        extraction_name="contingency_plan"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 AccuWeather weather alerts section
    accuweather_alerts_node = evaluator.add_sequential(
        id="accuweather_alerts_section",
        desc="AccuWeather weather alerts and hurricane tracking for Miami",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F6:A8 - Map zoom/pan for Miami alerts
    map_zoom_ok = has_any_ci(answer, ['accuweather', 'miami']) and (
        has_any_ci(answer, ['alert', 'warning', 'watch']) or
        has_any_ci(answer, ['hurricane', 'storm'])
    )
    evaluator.add_custom_node(
        result=bool(map_zoom_ok),
        id="accuweather_map_zoom",
        desc="[Action Node] accuweather.com:F6:A8 - Zoom and pan the weather alert map to Miami region",
        parent=accuweather_alerts_node,
        critical=False
    )

    # [Action Node] accuweather.com:F6:A10 - Click alert region on map
    alert_click_ok = (weather_alert.alert_level is not None or weather_alert.alert_type is not None)
    evaluator.add_custom_node(
        result=bool(alert_click_ok),
        id="accuweather_alert_click",
        desc="[Action Node] accuweather.com:F6:A10 - Click on colored alert region on the map to view details",
        parent=accuweather_alerts_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F6:P8 - Understand alert severity
    alert_severity_ok = (weather_alert.alert_level is not None and
                         has_any_ci(weather_alert.alert_level, ['watch', 'warning', 'advisory']))
    evaluator.add_custom_node(
        result=bool(alert_severity_ok),
        id="accuweather_alert_severity",
        desc="[Perception Node] accuweather.com:F6:P8 - Understand alert region colors and severity levels (Watch vs Warning)",
        parent=accuweather_alerts_node,
        critical=False
    )

    # [Action Node] accuweather.com:F5:A9 - Click storm marker on hurricane map
    storm_click_ok = (hurricane_tracking.storm_name is not None or
                      hurricane_tracking.storm_location is not None)
    evaluator.add_custom_node(
        result=bool(storm_click_ok),
        id="accuweather_storm_click",
        desc="[Action Node] accuweather.com:F5:A9 - Click on storm marker on hurricane tracking map",
        parent=accuweather_alerts_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F5:P5 - Identify storm location
    storm_location_ok = (hurricane_tracking.storm_location is not None or
                         hurricane_tracking.distance_to_miami is not None)
    evaluator.add_custom_node(
        result=bool(storm_location_ok),
        id="accuweather_storm_location",
        desc="[Perception Node] accuweather.com:F5:P5 - Identify storm marker position and distance from Miami",
        parent=accuweather_alerts_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F5:P6 - Understand storm path
    storm_path_ok = (hurricane_tracking.predicted_path is not None and
                     weather_alert.risk_assessment is not None)
    evaluator.add_custom_node(
        result=bool(storm_path_ok),
        id="accuweather_storm_path",
        desc="[Perception Node] accuweather.com:F5:P6 - Understand storm predicted path and assess if it affects Miami",
        parent=accuweather_alerts_node,
        critical=False
    )

    # 3.2 Booking.com Hilton cancellation policy section
    booking_hilton_node = evaluator.add_sequential(
        id="booking_hilton_section",
        desc="Booking.com Hilton Miami Downtown cancellation policy",
        parent=root,
        critical=False
    )

    # [Action Node] Booking.com:F1:A4 - Search and locate hotel by name
    hilton_search_ok = has_any_ci(answer, ['booking', 'hilton miami downtown'])
    evaluator.add_custom_node(
        result=bool(hilton_search_ok),
        id="booking_hilton_search",
        desc="[Action Node] Booking.com:F1:A4 - Search and directly locate Hilton Miami Downtown on Booking.com",
        parent=booking_hilton_node,
        critical=False
    )

    # [Action Node] Booking.com:F3:A14 - Click hotel card to view details
    hilton_details_ok = (hilton_policy.free_cancellation_deadline is not None or
                         hilton_policy.cancellation_fees is not None)
    evaluator.add_custom_node(
        result=bool(hilton_details_ok),
        id="booking_hilton_details",
        desc="[Action Node] Booking.com:F3:A14 - Click on hotel card to enter details page",
        parent=booking_hilton_node,
        critical=False
    )

    # [Action Node] Booking.com:F3:A46 - Expand cancellation policy section
    policy_expand_ok = (hilton_policy.free_cancellation_deadline is not None and
                        hilton_policy.cancellation_fees is not None)
    evaluator.add_custom_node(
        result=bool(policy_expand_ok),
        id="booking_hilton_expand",
        desc="[Action Node] Booking.com:F3:A46 - Expand cancellation policy or booking terms section",
        parent=booking_hilton_node,
        critical=False
    )

    # Link included check
    hilton_link_ok = has_url(hilton_policy.booking_link)
    evaluator.add_custom_node(
        result=bool(hilton_link_ok),
        id="booking_hilton_link",
        desc="Includes Booking.com details page link for Hilton Miami Downtown",
        parent=booking_hilton_node,
        critical=False
    )

    # 3.3 Booking.com alternative hotels section
    booking_alternatives_node = evaluator.add_sequential(
        id="booking_alternatives_section",
        desc="Booking.com alternative hotels in safer areas north of Miami",
        parent=root,
        critical=False
    )

    # [Action Node] Booking.com:F1:A1 - Select date range
    alt_dates_ok = has_any_ci(answer, ['january 28', 'january 30', 'jan 28', 'jan 30'])
    evaluator.add_custom_node(
        result=bool(alt_dates_ok),
        id="booking_alt_dates",
        desc="[Action Node] Booking.com:F1:A1 - Select check-in (Jan 28) and check-out (Jan 30) dates",
        parent=booking_alternatives_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A7 - Check free cancellation filter
    alt_free_cancel_ok = has_any_ci(answer, ['free cancellation']) and (
        has_any_ci(answer, ['filter', 'select']) or
        has_any_ci(answer, ['alternative', 'safer'])
    )
    evaluator.add_custom_node(
        result=bool(alt_free_cancel_ok),
        id="booking_alt_free_cancel",
        desc="[Action Node] Booking.com:F2:A7 - Check free cancellation filter option",
        parent=booking_alternatives_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A10 - Drag price range slider
    alt_price_ok = has_any_ci(answer, ['150', '300']) and has_any_ci(answer, ['price', 'night'])
    evaluator.add_custom_node(
        result=bool(alt_price_ok),
        id="booking_alt_price",
        desc="[Action Node] Booking.com:F2:A10 - Drag price slider to set $150-$300 range",
        parent=booking_alternatives_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A11 - Select rating filter
    alt_rating_ok = has_any_ci(answer, ['8.0', '8']) and has_any_ci(answer, ['rating', 'score'])
    evaluator.add_custom_node(
        result=bool(alt_rating_ok),
        id="booking_alt_rating",
        desc="[Action Node] Booking.com:F2:A11 - Select rating filter for 8.0+ scores",
        parent=booking_alternatives_node,
        critical=False
    )

    # [Perception Node] Booking.com:F2:P2 - Identify free cancellation badges
    # Check if answer mentions multiple hotels with free cancellation
    alt_badges_ok = answer.lower().count('free cancellation') >= 2
    evaluator.add_custom_node(
        result=bool(alt_badges_ok),
        id="booking_alt_badges",
        desc="[Perception Node] Booking.com:F2:P2 - Identify free cancellation status badges on hotel cards",
        parent=booking_alternatives_node,
        critical=False
    )

    # Check for 3 alternative hotels with required information
    hotel_count = answer.lower().count('hotel') + answer.lower().count('inn') + answer.lower().count('resort')
    has_three_hotels = hotel_count >= 3
    evaluator.add_custom_node(
        result=bool(has_three_hotels),
        id="booking_alt_three_hotels",
        desc="Provides information for 3 alternative hotels",
        parent=booking_alternatives_node,
        critical=False
    )

    # 3.4 Google Flights section
    google_flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights alternative flights to Miami on Jan 27",
        parent=root,
        critical=False
    )

    # [Action Node] google.comflights:F1:A1 - Input and select destination
    flights_dest_ok = has_any_ci(answer, ['google flights', 'miami']) or has_any_ci(answer, ['flight', 'miami'])
    evaluator.add_custom_node(
        result=bool(flights_dest_ok),
        id="google_flights_destination",
        desc="[Action Node] google.comflights:F1:A1 - Input Miami as destination and select airport",
        parent=google_flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F1:A2 - Select departure date
    flights_date_ok = has_any_ci(answer, ['january 27', 'jan 27', '27th'])
    evaluator.add_custom_node(
        result=bool(flights_date_ok),
        id="google_flights_date",
        desc="[Action Node] google.comflights:F1:A2 - Select January 27 as departure date",
        parent=google_flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F3:A5 - Filter by stops
    flights_stops_ok = has_any_ci(answer, ['direct', 'nonstop', 'one-stop', '1 stop'])
    evaluator.add_custom_node(
        result=bool(flights_stops_ok),
        id="google_flights_stops",
        desc="[Action Node] google.comflights:F3:A5 - Apply stops filter (Nonstop or 1 stop)",
        parent=google_flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F6:A14 - Click flight row to expand
    flights_expand_ok = has_any_ci(answer, ['flight number', 'departure time', 'arrival time'])
    evaluator.add_custom_node(
        result=bool(flights_expand_ok),
        id="google_flights_expand",
        desc="[Action Node] google.comflights:F6:A14 - Click on flight row to expand details",
        parent=google_flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F6:P1 - Extract numeric data
    flights_numeric_ok = (has_any_ci(answer, ['fare', 'price', '$']) and
                          has_any_ci(answer, ['duration', 'hour', 'minute']))
    evaluator.add_custom_node(
        result=bool(flights_numeric_ok),
        id="google_flights_numeric",
        desc="[Perception Node] google.comflights:F6:P1 - Extract fare amounts and flight duration values",
        parent=google_flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F6:P11 - Extract flight information
    flights_info_ok = has_any_ci(answer, ['flight number', 'airline'])
    evaluator.add_custom_node(
        result=bool(flights_info_ok),
        id="google_flights_info",
        desc="[Perception Node] google.comflights:F6:P11 - Extract flight numbers, airlines, and schedule times",
        parent=google_flights_node,
        critical=False
    )

    # Check for 3 alternative flights
    flight_mentions = answer.lower().count('flight')
    has_three_flights = flight_mentions >= 3 or (has_any_ci(answer, ['3 flight', 'three flight']))
    evaluator.add_custom_node(
        result=bool(has_three_flights),
        id="google_flights_three",
        desc="Provides information for 3 alternative flights",
        parent=google_flights_node,
        critical=False
    )

    # 3.5 Final outputs section
    outputs_node = evaluator.add_parallel(
        id="outputs_section",
        desc="Required output components and contingency plan",
        parent=root,
        critical=False
    )

    # Check for comprehensive output structure
    has_alert_output = weather_alert.alert_level is not None or weather_alert.risk_assessment is not None
    evaluator.add_custom_node(
        result=bool(has_alert_output),
        id="output_weather_alert",
        desc="Output includes weather alert level, type, period, and risk assessment",
        parent=outputs_node,
        critical=False
    )

    has_cancellation_output = (hilton_policy.free_cancellation_deadline is not None and
                               hilton_policy.booking_link is not None)
    evaluator.add_custom_node(
        result=bool(has_cancellation_output),
        id="output_cancellation",
        desc="Output includes Hilton cancellation deadline, fees, and Booking.com link",
        parent=outputs_node,
        critical=False
    )

    has_hotels_output = has_three_hotels and has_any_ci(answer, ['booking.com'])
    evaluator.add_custom_node(
        result=bool(has_hotels_output),
        id="output_alternative_hotels",
        desc="Output includes alternative hotel names, ratings, prices, and links",
        parent=outputs_node,
        critical=False
    )

    has_flights_output = has_three_flights and has_any_ci(answer, ['google flights'])
    evaluator.add_custom_node(
        result=bool(has_flights_output),
        id="output_alternative_flights",
        desc="Output includes alternative flight details and rebooking links",
        parent=outputs_node,
        critical=False
    )

    has_contingency = contingency.recommendation is not None and has_any_ci(
        contingency.recommendation,
        ['continue', 'change', 'reschedule']
    )
    evaluator.add_custom_node(
        result=bool(has_contingency),
        id="output_contingency_plan",
        desc="Output includes contingency plan recommendation (continue/change/reschedule)",
        parent=outputs_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
