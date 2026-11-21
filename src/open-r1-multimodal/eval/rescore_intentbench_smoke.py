"""Re-score archived smoke outputs without importing the inference entry point."""

import argparse
import json
from pathlib import Path
import tarfile

from answer_scoring import score_response, summarize_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with tarfile.open(args.backup) as backup:
        original = json.load(backup.extractfile("eval_results/ib_intentbench_smoke_v2.json"))
        audited = json.load(backup.extractfile("eval_results/ib_intentbench_smoke_v2_audited.json"))
        questions = json.load(backup.extractfile("IntentBench/qa_smoke.json"))

    results = []
    for original_row, audited_row, question in zip(
        original["results"], audited["results"], questions, strict=True
    ):
        if (
            original_row["raw"] != question
            or audited_row["raw"] != question
            or original_row["output"] != audited_row["output"]
        ):
            raise ValueError("Archived results and questions do not align")

        score = score_response(
            original_row["output"], original_row["solution"], original_row["problem_type"]
        )
        results.append({
            **original_row,
            "original_prediction": original_row["prediction"],
            "original_reward": original_row["reward"],
            "previous_audited_prediction": audited_row["prediction"],
            "previous_audited_reward": audited_row["reward"],
            **score,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"results": results, "final_acc": [summarize_scores(results)]},
                  output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
