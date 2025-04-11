from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import re


class Metric(ABC):
    """
    Абстрактный базовый класс для метрик оценки ответов модели.
    """

    @abstractmethod
    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вычисляет метрику на основе результатов оценки.

        Args:
            results (List[Dict[str, Any]]): Список результатов оценки ответов модели.
                Каждый элемент содержит как минимум ключи 'parsed_answer' и 'expected_answer'.

        Returns:
            Dict[str, Any]: Результат вычисления метрики.
        """
        pass


class AccuracyMetric(Metric):
    """
    Метрика точности (accuracy) - доля правильных ответов.
    """

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вычисляет точность на основе результатов оценки.

        Args:
            results (List[Dict[str, Any]]): Список результатов с полем 'is_correct'.

        Returns:
            Dict[str, Any]: Словарь с результатами:
                - total_examples: общее количество примеров
                - correct_answers: количество правильных ответов
                - accuracy: доля правильных ответов
        """
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
    """
    Метрика точности по доменам (областям) заданий.
    """

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вычисляет точность по доменам на основе результатов оценки.

        Args:
            results (List[Dict[str, Any]]): Список результатов с полями 'is_correct' и 'domain'.

        Returns:
            Dict[str, Any]: Словарь с результатами по доменам.
        """
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
    """
    Метрика точного соответствия ответа модели и ожидаемого ответа.
    """

    def __init__(self, case_sensitive: bool = True, normalize: bool = False):
        """
        Инициализирует метрику точного соответствия.

        Args:
            case_sensitive (bool): Учитывать ли регистр при сравнении ответов.
            normalize (bool): Применять ли нормализацию к ответам перед сравнением
                             (удаление лишних пробелов, знаков препинания и т.д.).
        """
        self.case_sensitive = case_sensitive
        self.normalize = normalize

    def _normalize_text(self, text: str) -> str:
        """
        Нормализует текст для сравнения.

        Args:
            text (str): Исходный текст.

        Returns:
            str: Нормализованный текст.
        """
        if not self.case_sensitive:
            text = text.lower()

        if self.normalize:
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вычисляет метрику точного соответствия.

        Args:
            results (List[Dict[str, Any]]): Список результатов.

        Returns:
            Dict[str, Any]: Результат вычисления метрики.
        """
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
    """
    Метрика F1-меры для бинарной классификации.
    """

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вычисляет F1-меру для бинарной классификации.

        Args:
            results (List[Dict[str, Any]]): Список результатов с полями 'is_correct' и 'expected_answer'.

        Returns:
            Dict[str, Any]: Словарь с результатами:
                - precision: точность (precision)
                - recall: полнота (recall)
                - f1_score: F1-мера
                - true_positives: количество истинно положительных результатов
                - false_positives: количество ложно положительных результатов
                - false_negatives: количество ложно отрицательных результатов
        """
        if not results:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "total_examples": 0,
            }

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for result in results:
            parsed_answer = result.get("parsed_answer", "")
            expected_answer = result.get("expected_answer", "")
            is_correct = result.get("is_correct", False)

            # Обработка случая, когда ответы должны совпадать
            if expected_answer == parsed_answer:
                if is_correct:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if is_correct:
                    false_positives += 1

        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives > 0
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_examples": len(results),
        }


class CompositeMetric(Metric):
    """
    Комбинированная метрика, которая применяет несколько метрик и объединяет их результаты.
    """

    def __init__(self, metrics: List[Metric], metric_names: Optional[List[str]] = None):
        """
        Инициализирует комбинированную метрику.

        Args:
            metrics (List[Metric]): Список метрик для применения.
            metric_names (Optional[List[str]]): Имена метрик для использования в качестве префиксов.
                                             Если не указаны, используются имена классов.
        """
        self.metrics = metrics

        # Автоматически генерируем имена метрик на основе их классов, если не указаны
        if metric_names is None:
            self.metric_names = [self._get_metric_name(metric) for metric in metrics]
        else:
            self.metric_names = metric_names

        if len(self.metrics) != len(self.metric_names):
            raise ValueError(
                "Количество метрик должно совпадать с количеством имен метрик"
            )

    def _get_metric_name(self, metric: Metric) -> str:
        """
        Генерирует имя метрики на основе её класса.

        Args:
            metric (Metric): Метрика.

        Returns:
            str: Имя метрики.
        """
        class_name = metric.__class__.__name__
        return class_name.replace("Metric", "").lower()

    def calculate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вычисляет все метрики и объединяет их результаты.

        Args:
            results (List[Dict[str, Any]]): Список результатов.

        Returns:
            Dict[str, Any]: Объединенные результаты всех метрик.
        """
        combined_results = {}

        for i, (metric, name) in enumerate(zip(self.metrics, self.metric_names)):
            metric_results = metric.calculate(results)

            prefix = "" if i == 0 else f"{name}_"

            for key, value in metric_results.items():
                combined_results[f"{prefix}{key}"] = value

        return combined_results
