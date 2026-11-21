import unittest

from answer_scoring import score_response, summarize_scores


class AnswerScoringTests(unittest.TestCase):
    def test_single_choice(self):
        self.assertEqual(score_response("<answer>A</answer>", "<answer>A</answer>",
                                        "multiple choice"),
                         {"prediction": "A", "prediction_status": "valid", "reward": 1.0})

    def test_judge_answers(self):
        for answer in ("Yes", "No"):
            with self.subTest(answer=answer):
                score = score_response(f"<answer>{answer.lower()}</answer>",
                                       f"<answer>{answer}</answer>", "judge")
                self.assertEqual(score["prediction"], answer)
                self.assertEqual(score["reward"], 1.0)

    def test_judge_rejects_substrings_and_ambiguity(self):
        for answer in ("yesterday", "nobody", "Yes or No", "No, but yes"):
            with self.subTest(answer=answer):
                score = score_response(f"<answer>{answer}</answer>", "<answer>No</answer>",
                                       "judge")
                self.assertEqual(score["prediction_status"], "invalid_answer_format")
                self.assertEqual(score["reward"], 0.0)

    def test_invalid_answers(self):
        examples = {
            "B": "missing_answer",
            "<answer> \n </answer>": "empty_answer",
            "<answer>B": "unclosed_answer",
            "<answer>B</ans": "unclosed_answer",
            "<answer>B or C</answer>": "invalid_answer_format",
            "<answer>B</answer><answer>C</answer>": "invalid_answer_tags",
        }
        for response, status in examples.items():
            with self.subTest(response=response):
                self.assertEqual(score_response(response, "<answer>B</answer>",
                                                "multiple choice"),
                                 {"prediction": "", "prediction_status": status, "reward": 0.0})

    def test_predictions_do_not_leak_between_rows(self):
        responses = ["<answer>B</answer>", "<think>Working", "<answer>C</answer>"]
        rows = [score_response(response, "<answer>B</answer>", "multiple choice")
                for response in responses]
        self.assertEqual([row["prediction"] for row in rows], ["B", "", "C"])
        self.assertEqual([row["reward"] for row in rows], [1.0, 0.0, 0.0])

    def test_multiselect_f1_and_mean(self):
        rows = [
            score_response("<answer>B,E</answer>", "<answer>E,B</answer>", "emer_ov_mc"),
            score_response("<answer>B</answer>", "<answer>B,E</answer>", "emer_ov_mc"),
            score_response("B,E", "<answer>B,E</answer>", "emer_ov_mc"),
        ]
        self.assertEqual(rows[0]["prediction"], "B,E")
        self.assertEqual([row["reward"] for row in rows], [1.0, 2 / 3, 0.0])
        self.assertAlmostEqual(summarize_scores(rows)["mean_acc"], 5 / 9)


if __name__ == "__main__":
    unittest.main()
