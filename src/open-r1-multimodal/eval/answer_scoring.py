"""Parse and score saved model responses independently of model inference."""

import math
import re


def parse_answer(response, question_type):
    tags = re.findall(r"</?answer\b[^<>]*>", response, flags=re.IGNORECASE)
    if not tags:
        return "", "missing_answer"
    if len(tags) == 1 and tags[0] == "<answer>":
        return "", "unclosed_answer"
    if tags != ["<answer>", "</answer>"]:
        return "", "invalid_answer_tags"

    match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if match is None:
        return "", "invalid_answer_tags"
    answer = match.group(1).strip()
    if not answer:
        return "", "empty_answer"
    if "<" in answer or ">" in answer:
        return "", "invalid_answer_format"

    if question_type == "multiple choice" and not re.fullmatch(r"[A-Z]", answer):
        return "", "invalid_answer_format"
    if question_type == "emer_ov_mc":
        if not re.fullmatch(r"[A-Z](?:\s*,\s*[A-Z])*", answer):
            return "", "invalid_answer_format"
        options = [option.strip() for option in answer.split(",")]
        if len(options) != len(set(options)):
            return "", "invalid_answer_format"
        answer = ",".join(options)
    if question_type == "judge":
        if answer.lower() not in ("yes", "no"):
            return "", "invalid_answer_format"
        answer = answer.capitalize()
    if question_type in ("numerical", "regression"):
        number = normalize_number(answer)
        if number is None or not math.isfinite(number):
            return "", "invalid_answer_format"
    return answer, "valid"


def score_response(response, solution, question_type):
    prediction, status = parse_answer(response, question_type)
    target, target_status = parse_answer(solution, question_type)
    reward = 0.0
    if status == target_status == "valid":
        reward = reward_fn(prediction, target, question_type)
    return {"prediction": prediction, "prediction_status": status, "reward": reward}


def summarize_scores(results):
    return {"mean_acc": sum(row["reward"] for row in results) / len(results) if results else 0.0}


def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

    import torch

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)

    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


def emer_ov_mc(reference, hypothesis):
    list_a = reference.split(",")
    list_b = hypothesis.split(",")
    true_positive = len(set(list_a) & set(list_b))
    precision = true_positive / len(list_a) if list_a else 0
    recall = true_positive / len(list_b) if list_b else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score

def reward_fn(output_ans, gt_ans, question_type):
    try:

        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        elif question_type == "emer_ov_mc":
            return emer_ov_mc(output_ans, gt_ans)
        elif  question_type == "judge":
            return float(output_ans == gt_ans)

        else:
            return 0.0
    except Exception as e:
        return 0.0
