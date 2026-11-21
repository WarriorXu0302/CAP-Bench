"""Exercise the scoring tree with fixed extraction output and no network access."""
import asyncio
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from cap_eval.evaluator import Evaluator


SCRIPTS = Path(__file__).resolve().parents[1] / "eval_scripts" / "2026_11_20"


def evaluate(task_id, answer, **extractions):
    spec = importlib.util.spec_from_file_location(task_id, SCRIPTS / f"task-{task_id}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def extract(self, prompt, template_class, extraction_name):
        return template_class(**extractions.get(extraction_name, {}))

    with patch("cap_eval.evaluator.create_evaluator", return_value=(None, None)), \
            patch.object(Evaluator, "extract", extract):
        summary = asyncio.run(module.evaluate_answer(None, answer, "test", "test", None, None, None))

    nodes = {}
    def visit(node):
        nodes[node["id"]] = node
        for child in node["children"]:
            visit(child)

    visit(summary["eval_breakdown"][0]["verification_tree"])
    return nodes


class GraderConditionsTest(unittest.TestCase):
    def test_explicit_zero_stations_completes_task_without_yelp(self):
        note = "No qualifying Superchargers were found."
        answer = ("Car and Driver (https://caranddriver.com/test) tested 230 miles at 75 mph. "
                  "80% is 184 miles. Google Maps from Griffith Observatory to Bellagio, "
                  f"search along the route between 150 and 184 miles: {note}")
        nodes = evaluate("2f4ebf", answer, caranddriver_data={
            "tested_75mph_range_text": "230 miles", "safe_maximum_distance_text": "184 miles",
            "article_link": "https://caranddriver.com/test",
        }, supercharger_stations={"station_count_note": note, "qualifying_count_reported": 0,
                                  "searched_window_quote": "150 and 184 miles"})
        self.assertEqual(nodes["root"]["score"], 1)
        self.assertNotIn("yelp_section", nodes)
        self.assertNotIn("output_station_info", nodes)

    def test_zero_stations_is_not_assumed(self):
        for data in [{}, {"station_count_note": "could not check"},
                     {"station_count_note": "No stations found", "qualifying_count_reported": 0}]:
            with self.subTest(data=data):
                nodes = evaluate("2f4ebf", "I could not check.", supercharger_stations=data)
                self.assertIn("yelp_section", nodes)
                self.assertEqual(nodes["googlemaps_station_count"]["score"], 0)

    def test_wrong_or_unsupported_search_window_keeps_station_checks(self):
        note = "No qualifying Superchargers were found."
        for searched, supplied_quote in [
            ("100 to 150 miles", "100 to 150 miles"),
            ("150 to 160 miles", "150 to 160 miles"),
            ("150 to 184 km", "150 to 184 km"),
            ("100 to 150 miles", "150 to 184 miles"),
            ("an unspecified stretch", None),
        ]:
            with self.subTest(searched=searched, quote=supplied_quote):
                answer = ("Google Maps from Griffith Observatory to Bellagio, search along the route "
                          f"from {searched}. {note}")
                nodes = evaluate("2f4ebf", answer, caranddriver_data={"safe_maximum_distance_text": "184 miles"},
                                 supercharger_stations={"station_count_note": note, "qualifying_count_reported": 0,
                                                       "searched_window_quote": supplied_quote})
                self.assertEqual(nodes["googlemaps_distance_constraint"]["score"], 0)
                self.assertIn("yelp_section", nodes)
                self.assertLess(nodes["root"]["score"], 1)

    def test_named_upper_bound_matches_the_safe_maximum(self):
        note = "No qualifying Superchargers were found."
        window = "mile 150 to my safe limit"
        answer = f"On Google Maps from Griffith Observatory to Bellagio, searched {window}. {note}"
        nodes = evaluate("2f4ebf", answer, caranddriver_data={"safe_maximum_distance_text": "184 miles"},
                         supercharger_stations={"station_count_note": note, "qualifying_count_reported": 0,
                                               "searched_window_quote": window})
        self.assertEqual(nodes["googlemaps_distance_constraint"]["score"], 1)
        self.assertNotIn("output_station_info", nodes)

    def test_one_documented_station_keeps_yelp(self):
        note = "Only one qualifying Supercharger was found."
        nodes = evaluate("2f4ebf", "Google Maps, Griffith Observatory to Bellagio, "
                         f"between 150 and 184 miles. {note}", supercharger_stations={
                             "station_count_note": note, "qualifying_count_reported": 1,
                             "stations": [{"station_name": "A", "distance_from_origin_text": "162 miles"}]})
        self.assertEqual(nodes["googlemaps_station_count"]["score"], 1)
        self.assertIn("yelp_section", nodes)

    def test_paper_present_absent_inconclusive(self):
        for status, applicable in [(True, True), (False, False), (None, True)]:
            with self.subTest(status=status):
                answer = "I checked the model card."
                quote = "The model card contains no arXiv reference." if status is False else None
                if quote:
                    answer += " " + quote
                nodes = evaluate("72b9f2", answer, paper_info={
                    "card_references_arxiv": status, "no_reference_quote": quote})
                self.assertEqual("arxiv_action_navigate" in nodes, applicable)
                self.assertEqual("report_paper_title" in nodes, applicable)
                self.assertEqual("report_architecture" in nodes, applicable)
                self.assertEqual(nodes["arxiv_action_find_reference"]["score"], int(status is not None))
                self.assertIn("report_model_name", nodes)

    def test_quoted_paper_not_waived_by_contradictory_flag(self):
        nodes = evaluate("72b9f2", "The model card cites an arXiv paper.", paper_info={
            "card_references_arxiv": False, "paper_title": "Text-to-Video Model Architecture"})
        self.assertIn("arxiv_perception_architecture", nodes)

    def test_video_search_outcomes_are_independent(self):
        nodes = evaluate("bc96e7", "YouTube results. Bilibili had no relevant videos.",
                         video_info={"bilibili_found_relevant": False, "youtube_found_relevant": True,
                                     "bilibili_empty_search_quote": "Bilibili had no relevant videos.",
                                     "youtube_videos": [{"title": "Apartment tour", "link": "https://youtu.be/test"}]})
        self.assertIn("youtube_video_details", nodes)
        self.assertEqual(nodes["bilibili_search"]["score"], 1)
        self.assertNotIn("bilibili_playback", nodes)
        self.assertNotIn("bilibili_video_details", nodes)

    def test_no_videos_requires_a_search_on_each_platform(self):
        for bb_status, expected in [(False, 1), (None, 0)]:
            with self.subTest(bb_status=bb_status):
                answer = "YouTube found no relevant videos."
                bb_quote = "No relevant results on Bilibili." if bb_status is False else None
                if bb_quote:
                    answer += " " + bb_quote
                nodes = evaluate("bc96e7", answer, video_info={
                    "youtube_found_relevant": False, "youtube_empty_search_quote": "YouTube found no relevant videos.",
                    "bilibili_found_relevant": bb_status, "bilibili_empty_search_quote": bb_quote})
                self.assertEqual(nodes["youtube_section"]["score"], 1)
                self.assertEqual(nodes["bilibili_section"]["score"], expected)
                self.assertIn("apartments_section", nodes)
                self.assertIn("route_section", nodes)

    def test_unsupported_negative_extractions_keep_requirements(self):
        paper = evaluate("72b9f2", "I could not open the model card.", paper_info={
            "card_references_arxiv": False, "no_reference_quote": "The card lists no paper."})
        self.assertIn("report_paper_title", paper)
        self.assertEqual(paper["arxiv_action_find_reference"]["score"], 0)
        videos = evaluate("bc96e7", "The sites were inaccessible.", video_info={
            "youtube_found_relevant": False, "bilibili_found_relevant": False,
            "bilibili_empty_search_quote": "Bilibili had no videos."})
        for platform in ["youtube", "bilibili"]:
            self.assertEqual(videos[platform + "_search"]["score"], 0)
            self.assertIn(platform + "_video_details", videos)

    def test_miami_alternatives_match_their_respective_prerequisites(self):
        cases = [
            # Miami risk, allows cancellation, flights at risk, score hotels, score flights
            ("high", True, True, True, False),
            ("high", False, None, False, False),
            ("low", True, True, False, True),
            ("low", True, False, False, False),
            (None, None, None, True, True),
        ]
        for risk, policy, disruption, hotels, flights in cases:
            with self.subTest(risk=risk, policy=policy, disruption=disruption):
                nodes = evaluate("1533cc", "AccuWeather Miami: no weather alerts or tracked storm.",
                                 weather_alert_info={"local_risk": risk, "active_alert": False,
                                                     "flight_disruption_possible": disruption},
                                 hurricane_tracking_info={"relevant_storm": False},
                                 hilton_cancellation_policy={"cancellation_allowed": policy})
                self.assertEqual("booking_alternatives_section" in nodes, hotels)
                self.assertEqual("output_alternative_hotels" in nodes, hotels)
                self.assertEqual("google_flights_section" in nodes, flights)
                self.assertEqual("output_alternative_flights" in nodes, flights)
                self.assertNotIn("accuweather_alert_severity", nodes)
                self.assertNotIn("accuweather_storm_path", nodes)
                self.assertIn("booking_hilton_section", nodes)
                self.assertIn("output_contingency_plan", nodes)

    def test_unknown_or_active_weather_keeps_alert_and_tracking_checks(self):
        for outcome in [None, True]:
            with self.subTest(outcome=outcome):
                nodes = evaluate("1533cc", "", weather_alert_info={"active_alert": outcome},
                                 hurricane_tracking_info={"relevant_storm": outcome})
                self.assertIn("accuweather_alert_severity", nodes)
                self.assertIn("accuweather_storm_path", nodes)


if __name__ == "__main__":
    unittest.main()
