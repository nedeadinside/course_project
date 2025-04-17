import os
import sys
from pathlib import Path

project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.data.dataset_builder import DatasetBuilder
from src.data.converters import (
    MmluCsvToJsonlConverter,
    MmluProCsvToJsonlConverter,
    XLSumJsonlConverter,
)
from src.data.config import (
    MMLU_INSTRUCTION_TEMPLATE,
    RUSSIAN_SUMMARIZATION_TEMPLATE,
    ENGLISH_SUMMARIZATION_TEMPLATE,
)


DEFAULT_INSTRUCTION = (
    "You are presented with a multiple-choice question on {subject}.\n\n"
    "QUESTION:\n{text}\n\n"
    "ANSWER OPTIONS:\n{options}\n\n"
    "INSTRUCTIONS:\n"
    "- Provide your answer as a single letter in parentheses: (X)\n"
    "- Select only one correct answer\n"
    "- Do not include explanations or additional text\n\n"
    "Your answer:"
)


def main():
    """
    Основная функция для сборки наборов данных.
    """
    raw_data_dir = os.path.join(project_root, "data", "raw")
    processed_data_dir = os.path.join(project_root, "data", "processed")

    os.makedirs(processed_data_dir, exist_ok=True)

    builder = DatasetBuilder(processed_data_dir, DEFAULT_INSTRUCTION)

    builder.register_converter("mmlu_csv", MmluCsvToJsonlConverter)
    builder.register_converter("mmlu_pro_csv", MmluProCsvToJsonlConverter)
    builder.register_converter("xlsum_jsonl", XLSumJsonlConverter)

    # MMLU
    mmlu_input_path = os.path.join(raw_data_dir, "mmlu", "mmlu_all_test.csv")
    mmlu_output_dir = os.path.join(processed_data_dir, "mmlu")
    os.makedirs(mmlu_output_dir, exist_ok=True)

    if os.path.exists(mmlu_input_path):
        builder.add_dataset(
            name="mmlu",
            input_path=mmlu_input_path,
            converter_name="mmlu_csv",
            output_filename=os.path.join("mmlu", "mmlu.jsonl"),
            instruction=MMLU_INSTRUCTION_TEMPLATE,
        )

    # MMLU Pro
    mmlu_pro_dir = os.path.join(raw_data_dir, "mmlu_pro")
    mmlu_pro_output_dir = os.path.join(processed_data_dir, "mmlu_pro")
    os.makedirs(mmlu_pro_output_dir, exist_ok=True)

    if os.path.exists(mmlu_pro_dir):
        mmlu_pro_files = [
            os.path.join(mmlu_pro_dir, f)
            for f in os.listdir(mmlu_pro_dir)
            if f.endswith(".csv")
        ]

        if mmlu_pro_files:
            mmlu_pro_input_path = mmlu_pro_files[0]
            builder.add_dataset(
                name="mmlu_pro",
                input_path=mmlu_pro_input_path,
                converter_name="mmlu_pro_csv",
                output_filename=os.path.join("mmlu_pro", "mmlu_pro.jsonl"),
                instruction=MMLU_INSTRUCTION_TEMPLATE,
            )

    # XLSum - English
    xlsum_dir = os.path.join(raw_data_dir, "XLSum")
    xlsum_output_dir = os.path.join(processed_data_dir, "xlsum")
    os.makedirs(xlsum_output_dir, exist_ok=True)

    if os.path.exists(xlsum_dir):
        # Обработка английских данных XLSum
        english_xlsum_file = os.path.join(xlsum_dir, "english_test.jsonl")
        if os.path.exists(english_xlsum_file):
            builder.add_dataset(
                name="xlsum_english",
                input_path=english_xlsum_file,
                converter_name="xlsum_jsonl",
                output_filename=os.path.join("xlsum", "xlsum_english.jsonl"),
                instruction=ENGLISH_SUMMARIZATION_TEMPLATE,
            )

        # Обработка русских данных XLSum
        russian_xlsum_file = os.path.join(xlsum_dir, "russian_test.jsonl")
        if os.path.exists(russian_xlsum_file):
            builder.add_dataset(
                name="xlsum_russian",
                input_path=russian_xlsum_file,
                converter_name="xlsum_jsonl",
                output_filename=os.path.join("xlsum", "xlsum_russian.jsonl"),
                instruction=RUSSIAN_SUMMARIZATION_TEMPLATE,
            )

    print("Начинаем сборку наборов данных...")
    results = builder.build_all_datasets()

    all_success = all(results.values())

    if all_success:
        print("Все наборы данных успешно собраны!")
    else:
        failed_datasets = [name for name, success in results.items() if not success]
        print(
            f"Не удалось собрать следующие наборы данных: {', '.join(failed_datasets)}"
        )

    print("\nСтатистика наборов данных:")
    stats = builder.get_dataset_stats()

    for name, dataset_stats in stats.items():
        if dataset_stats["exists"]:
            print(f"- {name}:")
            print(f"  Путь: {dataset_stats['path']}")
            print(f"  Размер: {dataset_stats['size'] / 1024:.2f} КБ")
            print(f"  Записей: {dataset_stats['records']}")
        else:
            print(f"- {name}: не существует или не был собран")


if __name__ == "__main__":
    main()
