from typing import Dict, Any, List, Optional
import json
import os
from abc import ABC, abstractmethod

from src.evaluation.parsers import ResponseParser
from src.evaluation.metrics import Metric


class AbstractEvaluator(ABC):
    """
    Абстрактный базовый класс для оценщиков ответов модели.
    Определяет общий интерфейс и базовую функциональность для всех оценщиков.
    """

    def __init__(self, parser: ResponseParser, output_dir: Optional[str] = None):
        """
        Инициализирует базовый оценщик ответов.

        Args:
            parser (ResponseParser): Парсер для извлечения ответов из выходных данных модели.
            output_dir (str, optional): Директория для сохранения результатов оценки.
        """
        self.parser = parser
        self.output_dir = output_dir

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def change_parser(self, parser: ResponseParser) -> None:
        """
        Изменяет парсер для извлечения ответов из выходных данных модели.

        Args:
            parser (ResponseParser): Новый парсер для использования.
        """
        self.parser = parser

    @abstractmethod
    def evaluate_response(
        self, prompt: str, model_output: str, expected_output: str
    ) -> Dict[str, Any]:
        """
        Оценивает отдельный ответ модели.

        Args:
            prompt (str): Исходный промпт.
            model_output (str): Ответ модели.
            expected_output (str): Ожидаемый ответ (правильный ответ).

        Returns:
            Dict[str, Any]: Результаты оценки в виде словаря.
        """
        pass

    @abstractmethod
    def evaluate_dataset(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценивает результаты для целого набора данных.

        Args:
            results (List[Dict[str, Any]]): Список результатов, каждый из которых содержит
                                          промпт, ответ модели и ожидаемый ответ.

        Returns:
            Dict[str, Any]: Агрегированные результаты оценки.
        """
        pass

    def save_evaluation(
        self, evaluation_results: Dict[str, Any], filename: str
    ) -> None:
        """
        Сохраняет результаты оценки в файл.

        Args:
            evaluation_results (Dict[str, Any]): Результаты оценки для сохранения.
            filename (str): Имя файла для сохранения результатов.
        """
        if not self.output_dir:
            raise ValueError("Не указана директория для сохранения результатов")

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"Результаты оценки сохранены в {filepath}")


class Evaluator(AbstractEvaluator):
    """
    Базовый класс для оценки ответов модели.
    Реализует паттерн "Стратегия" с одним парсером и одной метрикой.
    """

    def __init__(
        self, parser: ResponseParser, metric: Metric, output_dir: Optional[str] = None
    ):
        """
        Инициализирует оценщик ответов.

        Args:
            parser (ResponseParser): Парсер для извлечения ответов из выходных данных модели.
            metric (Metric): Метрика для оценки качества ответов.
            output_dir (str, optional): Директория для сохранения результатов оценки.
        """
        super().__init__(parser, output_dir)
        self.metric = metric

    def change_metric(self, metric: Metric) -> None:
        """
        Изменяет метрику для оценки качества ответов.

        Args:
            metric (Metric): Новая метрика для использования.
        """
        self.metric = metric

    def evaluate_response(
        self, prompt: str, model_output: str, expected_output: str
    ) -> Dict[str, Any]:
        """
        Оценивает отдельный ответ модели.

        Args:
            prompt (str): Исходный промпт.
            model_output (str): Ответ модели.
            expected_output (str): Ожидаемый ответ (правильный ответ).

        Returns:
            Dict[str, Any]: Результаты оценки в виде словаря.
        """
        parsed_answer = self.parser.parse(model_output)
        expected_answer = expected_output.strip()

        # Определяем, является ли ответ правильным
        is_correct = parsed_answer == expected_answer

        return {
            "parsed_answer": parsed_answer,
            "expected_answer": expected_answer,
            "is_correct": is_correct,
            "prompt": prompt,
            "model_output": model_output,
        }

    def evaluate_dataset(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценивает результаты для целого набора данных.

        Args:
            results (List[Dict[str, Any]]): Список результатов, каждый из которых содержит
                                          промпт, ответ модели и ожидаемый ответ.

        Returns:
            Dict[str, Any]: Агрегированные результаты оценки.
        """
        # Фильтруем результаты с ошибками
        valid_results = [r for r in results if not r.get("error")]

        processed_results = []

        # Обрабатываем каждый результат индивидуально
        for result in valid_results:
            evaluation = self.evaluate_response(
                result["prompt"], result["model_output"], result["expected_output"]
            )

            # Создаем обработанный результат с необходимыми полями
            evaluation_with_meta = {
                "index": result.get("index"),
                "domain": result.get("domain", "unknown"),
                "parsed_answer": evaluation["parsed_answer"],
                "expected_answer": evaluation["expected_answer"],
                "is_correct": evaluation["is_correct"],
            }

            # Добавляем дополнительные метаданные, если они есть
            for key, value in result.items():
                if key not in ["prompt", "model_output", "expected_output", "error"]:
                    evaluation_with_meta[key] = value

            processed_results.append(evaluation_with_meta)

        # Применяем метрику для получения агрегированных результатов
        metric_results = self.metric.calculate(processed_results)

        # Добавляем детализированные результаты для возможности дальнейшего анализа
        metric_results["detailed_evaluations"] = processed_results

        return metric_results


class MultiMetricEvaluator(AbstractEvaluator):
    """
    Оценщик ответов модели, использующий один парсер и несколько метрик.
    Реализует паттерн "Стратегия" с возможностью комбинирования нескольких метрик.
    """

    def __init__(
        self,
        parser: ResponseParser,
        metrics: List[Metric],
        metric_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Инициализирует оценщик с несколькими метриками.

        Args:
            parser (ResponseParser): Парсер для извлечения ответов из выходных данных модели.
            metrics (List[Metric]): Список метрик для оценки качества ответов.
            metric_names (Optional[List[str]]): Имена метрик для использования в качестве префиксов.
                                             Если не указаны, используются имена классов.
            output_dir (str, optional): Директория для сохранения результатов оценки.
        """
        super().__init__(parser, output_dir)
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

    def change_metrics(
        self, metrics: List[Metric], metric_names: Optional[List[str]] = None
    ) -> None:
        """
        Изменяет набор метрик для оценки качества ответов.

        Args:
            metrics (List[Metric]): Новый список метрик для использования.
            metric_names (Optional[List[str]]): Имена метрик для использования в качестве префиксов.
                                             Если не указаны, используются имена классов.
        """
        self.metrics = metrics

        # Обновляем имена метрик
        if metric_names is None:
            self.metric_names = [self._get_metric_name(metric) for metric in metrics]
        else:
            self.metric_names = metric_names

        if len(self.metrics) != len(self.metric_names):
            raise ValueError(
                "Количество метрик должно совпадать с количеством имен метрик"
            )

    def add_metric(self, metric: Metric, metric_name: Optional[str] = None) -> None:
        """
        Добавляет новую метрику в список метрик.

        Args:
            metric (Metric): Новая метрика для добавления.
            metric_name (Optional[str]): Имя метрики. Если не указано, генерируется автоматически.
        """
        self.metrics.append(metric)

        if metric_name is None:
            self.metric_names.append(self._get_metric_name(metric))
        else:
            self.metric_names.append(metric_name)

    def remove_metric(self, metric_name: str) -> None:
        """
        Удаляет метрику из списка метрик по имени.

        Args:
            metric_name (str): Имя метрики для удаления.
        """
        if metric_name in self.metric_names:
            index = self.metric_names.index(metric_name)
            self.metrics.pop(index)
            self.metric_names.pop(index)
        else:
            raise ValueError(f"Метрика с именем '{metric_name}' не найдена")

    def evaluate_response(
        self, prompt: str, model_output: str, expected_output: str
    ) -> Dict[str, Any]:
        """
        Оценивает отдельный ответ модели.

        Args:
            prompt (str): Исходный промпт.
            model_output (str): Ответ модели.
            expected_output (str): Ожидаемый ответ (правильный ответ).

        Returns:
            Dict[str, Any]: Результаты оценки в виде словаря.
        """
        parsed_answer = self.parser.parse(model_output)
        expected_answer = expected_output.strip()

        # Определяем, является ли ответ правильным
        is_correct = parsed_answer == expected_answer

        return {
            "parsed_answer": parsed_answer,
            "expected_answer": expected_answer,
            "is_correct": is_correct,
            "prompt": prompt,
            "model_output": model_output,
        }

    def evaluate_dataset(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценивает результаты для целого набора данных используя все метрики.

        Args:
            results (List[Dict[str, Any]]): Список результатов, каждый из которых содержит
                                          промпт, ответ модели и ожидаемый ответ.

        Returns:
            Dict[str, Any]: Агрегированные результаты оценки.
        """
        # Фильтруем результаты с ошибками
        valid_results = [r for r in results if not r.get("error")]

        processed_results = []

        # Обрабатываем каждый результат индивидуально
        for result in valid_results:
            evaluation = self.evaluate_response(
                result["prompt"], result["model_output"], result["expected_output"]
            )

            # Создаем обработанный результат с необходимыми полями
            evaluation_with_meta = {
                "index": result.get("index"),
                "domain": result.get("domain", "unknown"),
                "parsed_answer": evaluation["parsed_answer"],
                "expected_answer": evaluation["expected_answer"],
                "is_correct": evaluation["is_correct"],
            }

            # Добавляем дополнительные метаданные, если они есть
            for key, value in result.items():
                if key not in ["prompt", "model_output", "expected_output", "error"]:
                    evaluation_with_meta[key] = value

            processed_results.append(evaluation_with_meta)

        # Применяем каждую метрику и объединяем результаты
        combined_results = {}

        for i, (metric, name) in enumerate(zip(self.metrics, self.metric_names)):
            metric_results = metric.calculate(processed_results)

            # Префикс для имен полей
            prefix = "" if i == 0 else f"{name}_"

            for key, value in metric_results.items():
                combined_results[f"{prefix}{key}"] = value

        combined_results["detailed_evaluations"] = processed_results

        return combined_results
