import unittest

from cap_eval.utils.numbers import extract_float, extract_int
from grader_test_utils import grade, task


class GraderValues(unittest.TestCase):
    def test_parse_first_number(self):
        for text, expected in [("$2,450.75 / month", 2450.75), ("-1,200.5 °C", -1200.5),
                               ("1,234,567 views", 1234567), ("4.5 out of 5", 4.5),
                               ("2, 5", 2), ("2000–2500", 2000), ("n/a", None), (None, None)]:
            with self.subTest(text=text):
                self.assertEqual(extract_float(text), expected)
        self.assertEqual(extract_int("12,450 reviews"), 12450)
        self.assertEqual(extract_int("2 guests, 5 beds"), 2)

    def test_task_budget_boundaries(self):
        austin = task("cdb0d8").looks_like_rent_in_range
        portland = task("c72408").is_valid_rent_range
        self.assertTrue(austin("$1,450 per month"))
        self.assertFalse(austin("$1,800.01"))
        self.assertTrue(portland("$2,450"))
        self.assertFalse(portland("$1,450"))
        self.assertFalse(portland("$-2,450"))

    def test_count_named_recommendations(self):
        module = task("c72408")
        for names, expected in [(["Laurel Court", "River Lofts"], 1), (["Two Rivers"], 0),
                                (["Rose 2", " ROSE  2 "], 0), ([], 0),
                                (["Laurel", "River", "Pine"], 0)]:
            with self.subTest(names=names):
                result = grade(module, ", ".join(names), recommendations={"recommended_apartments": names})
                self.assertEqual(result.find_node("two_apartments_recommended").score, expected)

    def test_getty_dates(self):
        check = task("26914d").looks_like_5th_4th_century_bce
        for text in ["about 470 B.C.", "ca. 500 B.C.E.", "450–400 BCE", "301 BC",
                     "late 5th–early 4th century B.C.", "4th century BCE", "circa 470", "5th-4th century"]:
            self.assertTrue(check(text), text)
        for text in ["501 BCE", "c. 300 B.C.", "350–275 BCE", "470 A.D.", "470 BCE–20 CE",
                     "4th–3rd century BCE", "circa 470 CE", "470", None]:
            self.assertFalse(check(text), text)


if __name__ == "__main__":
    unittest.main()
