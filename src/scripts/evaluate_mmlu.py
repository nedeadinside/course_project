import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.client.model_client import BatchModelClient
from src.prompts.prompt_generators import FewShotPromptGenerator
from src.prompts.prompt_strategies import OptionsPromptStrategy
from src.evaluation.evaluator import Evaluator
from src.evaluation.parsers import MultipleChoiceParser
from src.evaluation.metrics import AccuracyMetric, DomainAccuracyMetric, CompositeMetric


def parse_arguments():
    """
    Парсит аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description="Оценка модели на наборе данных MMLU")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Хост, где запущен сервер с моделью",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт, на котором слушает сервер",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="/api/v1/generate",
        help="Эндпоинт для запросов к модели",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Размер пакета для запросов",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Максимальное количество токенов в ответе",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "results", "evaluations"),
        help="Директория для сохранения результатов",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    processed_data_dir = os.path.join(project_root, "data", "processed")
    mmlu_data_path = os.path.join(processed_data_dir, "mmlu", "mmlu.jsonl")

    if not os.path.exists(mmlu_data_path):
        print(f"Ошибка: Файл данных MMLU не найден по пути: {mmlu_data_path}")
        print(
            "Пожалуйста, сначала запустите скрипт build_datasets.py для подготовки данных."
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Инициализация клиента для запросов к модели на {args.host}:{args.port}...")

    client = BatchModelClient(
        host=args.host,
        port=args.port,
        endpoint=args.endpoint,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    prompt_strategy = OptionsPromptStrategy()
    prompt_generator = FewShotPromptGenerator(strategy=prompt_strategy, n_shots=3)

    print(f"Загрузка данных MMLU из файла: {mmlu_data_path}")
    prompt_generator.load_data(mmlu_data_path)

    print("Инициализация парсера и метрик для оценки ответов модели...")

    parser = MultipleChoiceParser(case_sensitive=False)

    accuracy_metric = AccuracyMetric()
    domain_accuracy_metric = DomainAccuracyMetric()

    composite_metric = CompositeMetric(
        metrics=[accuracy_metric, domain_accuracy_metric],
        metric_names=["accuracy", "domain"],
    )

    evaluator = Evaluator(
        parser=parser, metric=composite_metric, output_dir=args.output_dir
    )

    print(f"Начинаем оценку модели на наборе данных MMLU...")
    print(f"Параметры: batch_size={args.batch_size}, max_tokens={args.max_tokens}")

    results = client.process_dataset(generator=prompt_generator)

    print(f"Обработка завершена. Получено {len(results)} результатов.")

    evaluation_results = evaluator.evaluate_dataset(results)

    print("\nРезультаты оценки:")
    print(f"Всего примеров: {evaluation_results['total_examples']}")
    print(f"Правильных ответов: {evaluation_results['correct_answers']}")
    print(f"Точность (accuracy): {evaluation_results['accuracy']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = f"mmlu_evaluation_{timestamp}.json"

    evaluator.save_evaluation(evaluation_results, results_filename)

    print(
        f"\nРезультаты сохранены в файл: {os.path.join(args.output_dir, results_filename)}"
    )


# python3 src/scripts/evaluate_mmlu.py --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    main()
