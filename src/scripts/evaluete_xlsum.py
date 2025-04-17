import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.client.model_client import BatchModelClient
from src.prompts.prompt_generators import (
    SinglePromptGenerator,
)
from src.prompts.prompt_strategies import (
    GenerationPromptStrategy,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.parsers import RegexParser
from src.evaluation.metrics import (
    BLEUMetric,
    ROUGEMetric,
    CompositeMetric,
)


def parse_arguments():
    """
    Парсит аргументы командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Оценка модели на наборе данных XLSum (суммаризация)"
    )

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
        "--language",
        type=str,
        default="russian",
        choices=["russian", "english"],
        help="Язык набора данных XLSum для оценки",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Размер пакета для запросов",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
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
    xlsum_data_path = os.path.join(
        processed_data_dir, "xlsum", f"xlsum_{args.language}.jsonl"
    )

    if not os.path.exists(xlsum_data_path):
        print(
            f"Ошибка: Файл данных XLSum ({args.language}) не найден по пути: {xlsum_data_path}"
        )
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

    prompt_strategy = GenerationPromptStrategy()
    prompt_generator = SinglePromptGenerator(strategy=prompt_strategy, n_shots=0)
    prompt_generator.load_data(xlsum_data_path)

    print("Инициализация парсера и метрик для оценки ответов модели...")

    parser = RegexParser(pattern=r"(.*)", group=1)

    bleu_metric = BLEUMetric(language=args.language)
    rouge_metric = ROUGEMetric()

    composite_metric = CompositeMetric(
        metrics=[bleu_metric, rouge_metric],
        metric_names=["bleu", "rouge"],
    )

    evaluator = Evaluator(
        parser=parser, metric=composite_metric, output_dir=args.output_dir
    )

    print(f"Начинаем оценку модели на наборе данных XLSum ({args.language})...")
    print(f"Параметры: batch_size={args.batch_size}, max_tokens={args.max_tokens}")

    results = client.process_dataset(
        generator=prompt_generator,
        max_tokens=args.max_tokens,
    )

    evaluation_results = evaluator.evaluate_dataset(results)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = f"xlsum_{args.language}_evaluation_{timestamp}.json"

    evaluator.save_evaluation(evaluation_results, results_filename)

    print(
        f"\nРезультаты сохранены в файл: {os.path.join(args.output_dir, results_filename)}"
    )


# python src/scripts/evaluate_xlsum.py --language russian --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    main()
