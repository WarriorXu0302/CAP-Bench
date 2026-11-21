import asyncio
import json
from tempfile import TemporaryDirectory
from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

from cap_eval.eval_runner import evaluate_task
from cap_eval.evaluator import Evaluator
from cap_eval.utils.grading_dates import coming_friday, months_before, on_itinerary_day, publication_in_window
from cap_eval.utils.load_eval_script import load_eval_script


SCRIPTS = Path(__file__).resolve().parents[1] / "eval_scripts" / "2026_11_20"


def date_leaves(task, reference, answer="", **extractions):
    evaluate = load_eval_script(SCRIPTS / f"task-{task}.py")

    async def extract(self, prompt, template_class, extraction_name):
        return template_class(**extractions.get(extraction_name, {}))

    def leaf_scores(evaluator):
        # Inspect the actual predicates before unrelated sequential prerequisites
        # suppress them in an intentionally partial answer.
        return {n.id: n.score for n in evaluator._iter_all_nodes() if not n.children}

    with patch("cap_eval.evaluator.create_evaluator", return_value=(None, None)), \
            patch.object(Evaluator, "extract", extract), patch.object(Evaluator, "get_summary", leaf_scores):
        return asyncio.run(evaluate(None, answer, "test", "test", None, None, None, evaluation_date=reference))


class GraderDatesTest(unittest.TestCase):
    def test_this_year_depends_on_reference_date(self):
        for ref, expected in [(date(2025, 12, 31), 0), (date(2026, 5, 1), 1), (date(2027, 5, 1), 0)]:
            with self.subTest(ref=ref):
                scores = date_leaves("5ea2c0", ref, "YouTube filter", youtube_video_info={
                    "video_publication_date": "April 22, 2026", "video_duration": "8:00"})
                self.assertEqual(scores["youtube_filters"], expected)

    def test_releases_cover_last_six_calendar_months(self):
        ref = date(2026, 3, 31)
        self.assertEqual(months_before(ref, 6), date(2025, 9, 30))
        for release, expected in [
            ("Sep 29, 2025", 0), ("Sep. 30, 2025", 1), ("2025-10", 1),
            ("March 31st, 2026", 1), ("4/1/2026", 0), ("2026", 0), ("2026-02-30", 0),
            ("2026-00-10", 0), ("2026-01-00", 0)
        ]:
            with self.subTest(release=release):
                scores = date_leaves("ac39b0", ref, "Recent releases", all_disc_data={
                    "criterion_discs": [{"release_date": release}]})
                self.assertEqual(scores["criterion_filter_recent"], expected)

    def test_any_past_video_publication_year(self):
        videos = {f"video_{n}": {"publication_date": value} for n, value in enumerate(
            ["2018-02-20", "2024年11月2日", "2 weeks ago"], start=1)}
        ref = date(2027, 1, 2)
        scores = date_leaves("73c4b7", ref, bilibili_videos=videos)
        self.assertEqual(scores["bilibili_video_publication_dates"], 1)
        self.assertNotIn("bilibili_action_date_sort", scores)
        videos["video_1"] = {"publication_date": "2027-01-03"}
        self.assertEqual(date_leaves("73c4b7", ref, bilibili_videos=videos)["bilibili_video_publication_dates"], 0)

    def test_itinerary_across_new_year_and_early_festival_end(self):
        ref = date(2027, 12, 30)
        fields = {
            "festival_info": {"start_time": "Friday at 5 pm", "end_time": "Jan 1, 2028 9:30 pm"},
            "outbound_flight": {"departure_time": "12/31/2027"},
            "return_flight": {"departure_time": "Sunday morning"},
            "accommodation": {"airbnb_link": "https://www.airbnb.com/rooms/1?check_in=2027-12-31&check_out=2028-01-03"},
        }
        scores = date_leaves("3e6e24", ref, **fields)
        for node in ["eventbrite_date_selector", "flights_date_selection", "airbnb_date_selection"]:
            self.assertEqual(scores[node], 1)
        fields["return_flight"]["departure_time"] = "Monday morning"
        fields["accommodation"]["airbnb_link"] = "https://www.airbnb.com/rooms/1?check_in=2027-12-31"
        scores = date_leaves("3e6e24", ref, "check-in / check-out", **fields)
        self.assertEqual(scores["flights_date_selection"], 0)
        self.assertEqual(scores["airbnb_date_selection"], 0)

    def test_runner_persists_date_and_accepts_legacy_signature(self):
        async def extract(self, prompt, template_class, extraction_name):
            return template_class()

        with TemporaryDirectory() as tmp, patch("cap_eval.evaluator.create_evaluator", return_value=(None, None)), \
                patch.object(Evaluator, "extract", extract):
            root, task = Path(tmp), "task-5ea2c0"
            answers = root / "answers" / "test" / task
            answers.mkdir(parents=True)
            (answers / "answer_1.md").write_text("test", encoding="utf-8")
            for version, reference in [
                ("2026_11_20", date(2027, 1, 2)), ("2026_11_20", date(2025, 8, 9)),
                ("2026_10_07", date(2025, 8, 9)),
            ]:
                script = SCRIPTS.parent / version / f"{task}.py"
                results = asyncio.run(evaluate_task(
                    None, task, "test", root / "answers", root / "cache", root / "output", script,
                    evaluation_date=reference,
                ))
                self.assertEqual((results[0]["eval_version"], results[0]["evaluation_date"]),
                                 (version, reference.isoformat()))
            files = list((root / "output").rglob("*/results/*.json"))
            self.assertEqual(len(files), 3)
            self.assertTrue(all("evaluation_date" in json.loads(p.read_text()) for p in files))

    def test_calendar_boundaries_and_precision(self):
        friday = date(2026, 11, 20)
        self.assertEqual(coming_friday(friday), friday)
        self.assertEqual(coming_friday(date(2026, 11, 21)), date(2026, 11, 27))
        self.assertTrue(on_itinerary_day("20 Nov 2026 at 9 AM", friday))
        self.assertTrue(on_itinerary_day("Nov 20", friday))
        self.assertFalse(on_itinerary_day("Friday, Nov 20, 2025", friday))
        self.assertFalse(on_itinerary_day("10:30 pm", friday))
        self.assertFalse(on_itinerary_day("last Friday", friday))
        self.assertFalse(publication_in_window("2026-12-01", date(2026, 1, 1), friday))
        self.assertFalse(publication_in_window("unknown", date.min, friday))


if __name__ == "__main__":
    unittest.main()
