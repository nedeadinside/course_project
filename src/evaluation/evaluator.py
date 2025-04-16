from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
import os

from src.evaluation.parsers import ResponseParser
from src.evaluation.metrics import Metric


class AbstractEvaluator(ABC):
    def __init__(self, parser: ResponseParser, output_dir: Optional[str] = None):
        self.parser = parser
        self.output_dir = output_dir

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def change_parser(self, parser: ResponseParser) -> None:
        self.parser = parser

    @abstractmethod
    def evaluate_response(
        self, prompt: str, model_output: str, expected_output: str
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate_dataset(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

    def save_evaluation(
        self, evaluation_results: Dict[str, Any], filename: str
    ) -> None:
        if not self.output_dir:
            raise ValueError("Не указана директория для сохранения результатов")

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"Результаты оценки сохранены в {filepath}")


class Evaluator(AbstractEvaluator):
    def __init__(
        self, parser: ResponseParser, metric: Metric, output_dir: Optional[str] = None
    ):
        super().__init__(parser, output_dir)
        self.metric = metric

    def change_metric(self, metric: Metric) -> None:
        self.metric = metric

    def evaluate_response(
        self, prompt: str, model_output: str, expected_output: str
    ) -> Dict[str, Any]:
        parsed_answer = self.parser.parse(model_output)
        expected_answer = expected_output.strip()
        is_correct = parsed_answer == expected_answer

        return {
            "parsed_answer": parsed_answer,
            "expected_answer": expected_answer,
            "is_correct": is_correct,
            "prompt": prompt,
            "model_output": model_output,
        }

    def evaluate_dataset(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_results = [r for r in results if not r.get("error")]
        processed_results = []

        for result in valid_results:
            evaluation = self.evaluate_response(
                result["prompt"], result["model_output"], result["expected_output"]
            )

            evaluation_with_meta = {
                "index": result.get("index"),
                "domain": result.get("domain", "unknown"),
                "parsed_answer": evaluation["parsed_answer"],
                "expected_answer": evaluation["expected_answer"],
                "is_correct": evaluation["is_correct"],
                "model_output": evaluation["model_output"],
            }

            for key, value in result.items():
                if key not in ["prompt", "model_output", "expected_output", "error"]:
                    evaluation_with_meta[key] = value

            processed_results.append(evaluation_with_meta)

        metric_results = self.metric.calculate(processed_results)
        metric_results["detailed_evaluations"] = processed_results

        return metric_results
