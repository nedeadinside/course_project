from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Literal
import re

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize


class Metric(ABC):
    @abstractmethod
    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


class AccuracyMetric(Metric):
    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"total_examples": 0, "correct_answers": 0, "accuracy": 0.0}

        total_evaluated = len(results)
        correct_count = sum(1 for result in results if result.get("is_correct", False))

        return {
            "total_examples": total_evaluated,
            "correct_answers": correct_count,
            "accuracy": correct_count / total_evaluated if total_evaluated > 0 else 0.0,
        }


class DomainAccuracyMetric(Metric):
    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"domain_stats": {}}

        domain_stats = {}
        for result in results:
            domain = result.get("domain", "unknown")
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "correct": 0}

            domain_stats[domain]["total"] += 1
            if result.get("is_correct", False):
                domain_stats[domain]["correct"] += 1

        for domain in domain_stats:
            total = domain_stats[domain]["total"]
            correct = domain_stats[domain]["correct"]
            domain_stats[domain]["accuracy"] = correct / total if total > 0 else 0.0

        return {"domain_stats": domain_stats}


class ExactMatchMetric(Metric):
    def __init__(self, case_sensitive: bool = False, normalize: bool = True):
        self.case_sensitive = case_sensitive
        self.normalize = normalize

    def _normalize_text(self, text: str) -> str:
        if not self.case_sensitive:
            text = text.lower()

        if self.normalize:
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_evaluated = len(results)
        correct_count = 0

        for result in results:
            parsed_answer = result.get("parsed_answer", "")
            expected_answer = result.get("expected_answer", "")

            if self.normalize:
                parsed_answer = self._normalize_text(parsed_answer)
                expected_answer = self._normalize_text(expected_answer)
            elif not self.case_sensitive:
                parsed_answer = parsed_answer.lower()
                expected_answer = expected_answer.lower()

            is_correct = parsed_answer == expected_answer
            result["is_correct"] = is_correct

            if is_correct:
                correct_count += 1

        return {
            "total_examples": total_evaluated,
            "correct_answers": correct_count,
            "accuracy": correct_count / total_evaluated if total_evaluated > 0 else 0.0,
        }


class F1ScoreMetric(Metric):
    def __init__(self, pos_label=True):
        self.pos_label = pos_label

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
                "total_examples": 0,
            }

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for result in results:
            prediction = result.get("is_correct", False)
            expected = self.pos_label

            if prediction == expected == self.pos_label:
                true_positives += 1
            elif prediction == self.pos_label and expected != self.pos_label:
                false_positives += 1
            elif prediction != self.pos_label and expected == self.pos_label:
                false_negatives += 1
            else:
                true_negatives += 1

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "total_examples": len(results),
        }


class CompositeMetric(Metric):
    def __init__(self, metrics: List[Metric], metric_names: Optional[List[str]] = None):
        self.metrics = metrics

        if metric_names is None:
            self.metric_names = [self._get_metric_name(metric) for metric in metrics]
        else:
            self.metric_names = metric_names

        if len(self.metrics) != len(self.metric_names):
            raise ValueError(
                "Количество метрик должно совпадать с количеством имен метрик"
            )

    def _get_metric_name(self, metric: Metric) -> str:
        class_name = metric.__class__.__name__
        return class_name.replace("Metric", "").lower()

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined_results = {}

        for i, (metric, name) in enumerate(zip(self.metrics, self.metric_names)):
            metric_results = metric.calculate(results)

            prefix = "" if i == 0 else f"{name}_"

            for key, value in metric_results.items():
                combined_results[f"{prefix}{key}"] = value

        return combined_results


class BLEUMetric(Metric):
    SupportedLanguage = Literal["english", "russian"]

    def __init__(
        self,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        language: SupportedLanguage = "english",
    ):
        self.weights = weights
        if smoothing_function is None:
            self.smoothing_function = SmoothingFunction().method1
        else:
            self.smoothing_function = smoothing_function
        self.language = language

    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text.lower(), language=self.language)

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"bleu_score": 0.0, "total_examples": 0}

        total_score = 0.0
        bleu_scores = []

        for result in results:
            reference = result.get("expected_answer", "")
            hypothesis = result.get("parsed_answer", "")

            reference_tokens = self._tokenize(reference)
            hypothesis_tokens = self._tokenize(hypothesis)

            if len(reference_tokens) == 0 or len(hypothesis_tokens) == 0:
                score = 0.0
            else:
                score = sentence_bleu(
                    [reference_tokens],
                    hypothesis_tokens,
                    weights=self.weights,
                    smoothing_function=self.smoothing_function,
                )

            total_score += score
            bleu_scores.append(score)

        avg_score = total_score / len(results)

        return {
            "bleu_score": avg_score,
            "individual_scores": bleu_scores,
            "total_examples": len(results),
        }


class ROUGEMetric(Metric):
    def __init__(
        self,
        rouge_types=None,
        use_stemmer=True,
    ):
        if rouge_types is None:
            self.rouge_types = ["rouge1", "rouge2", "rougeL"]
        else:
            self.rouge_types = rouge_types

        self.scorer = rouge_scorer.RougeScorer(
            self.rouge_types, use_stemmer=use_stemmer
        )

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            result_dict = {}
            for rouge_type in self.rouge_types:
                result_dict[f"{rouge_type}_precision"] = 0.0
                result_dict[f"{rouge_type}_recall"] = 0.0
                result_dict[f"{rouge_type}_f1"] = 0.0
            result_dict["total_examples"] = 0
            return result_dict

        rouge_precision = {rouge_type: [] for rouge_type in self.rouge_types}
        rouge_recall = {rouge_type: [] for rouge_type in self.rouge_types}
        rouge_f1 = {rouge_type: [] for rouge_type in self.rouge_types}

        for result in results:
            reference = result.get("expected_answer", "")
            hypothesis = result.get("parsed_answer", "")

            if not reference or not hypothesis:
                continue

            scores = self.scorer.score(reference, hypothesis)

            for rouge_type in self.rouge_types:
                rouge_precision[rouge_type].append(scores[rouge_type].precision)
                rouge_recall[rouge_type].append(scores[rouge_type].recall)
                rouge_f1[rouge_type].append(scores[rouge_type].fmeasure)

        result_dict = {}
        for rouge_type in self.rouge_types:
            precisions = rouge_precision[rouge_type]
            avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
            result_dict[f"{rouge_type}_precision"] = avg_precision

            recalls = rouge_recall[rouge_type]
            avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
            result_dict[f"{rouge_type}_recall"] = avg_recall

            f1_scores = rouge_f1[rouge_type]
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            result_dict[f"{rouge_type}_f1"] = avg_f1

        result_dict["total_examples"] = len(results)

        return result_dict
