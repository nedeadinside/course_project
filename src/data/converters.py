from abc import ABCMeta, abstractmethod
import pandas as pd
import string
import re
import json
import os
from typing import Dict, Any, List


class DataConverter(metaclass=ABCMeta):
    """
    Абстрактный базовый класс для преобразования данных из одного формата в другой.
    Все конкретные конвертеры должны наследоваться от этого класса.
    """

    def __init__(self, input_path: str, output_path: str, instruction: str = None):
        """
        Инициализация конвертера.
        Args:
            input_path: Путь к входным данным
            output_path: Путь к выходным данным
            instruction: Инструкция для задачи (если применимо)
        """
        self.instruction = instruction
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def convert(self) -> bool:
        """
        Преобразовать данные из одного формата в другой.
        Returns:
            bool: True, если преобразование прошло успешно, иначе False
        """
        pass

    def validate_input(self) -> bool:
        """
        Проверить существование и доступность входных данных.
        Returns:
            bool: True, если входные данные существуют и доступны, иначе False
        """
        return os.path.exists(self.input_path)


class CsvToJsonlConverter(DataConverter):
    """
    Класс для преобразования данных из формата CSV в JSONL.
    """

    def write_jsonl(self, data: List[Dict[str, Any]]) -> bool:
        """
        Записать данные в формате JSONL.
        Args:
            data: Список словарей для записи в JSONL
        Returns:
            bool: True, если запись прошло успешно, иначе False
        """
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"Ошибка при записи JSONL: {e}")
            return False

    def preprocess_options(self, options_str):
        """
        Преобразует строку с опциями в список опций.
        Args:
            options_str: Строка с опциями в формате '["опция1", "опция2", ...]' или "['опция1', 'опция2', ...]"
        Returns:
            list: Список опций
        """
        clean_str = options_str.strip("[]").strip()

        pattern = r'(?:"([^"]*)")|(?:\'([^\']*)\')'
        matches = re.findall(pattern, clean_str)

        options = []
        for match in matches:
            option = match[0] if match[0] else match[1]
            if option.strip():
                options.append(option.strip())

        return options

    def get_letter_by_index(self, idx):
        """
        Преобразует числовой индекс в букву ответа (A, B, C, ...).
        Args:
            idx: Индекс опции ответа
        Returns:
            str: Буква, соответствующая индексу
        """
        return string.ascii_uppercase[idx]

    def create_options_data(self, options_list):
        """
        Создает словарь опций и текст с опциями.
        Args:
            options_list: Список опций
        Returns:
            tuple: (словарь с опциями, текстовое представление опций)
        """
        options_dict = {}
        options_text = ""
        for i, option in enumerate(options_list):
            letter = self.get_letter_by_index(i)
            options_text += f"{letter} {option}\n"
            options_dict[f"option_{letter.lower()}"] = option

        return options_dict, options_text.strip()

    @abstractmethod
    def process_row(self, row) -> Dict[str, Any]:
        """
        Обрабатывает строку данных и преобразует их в формат JSONL.
        Args:
            row: Строка данных из CSV файла
        Returns:
            dict: Запись в формате JSONL
        """
        pass


class MmluCsvToJsonlConverter(CsvToJsonlConverter):
    """Конвертер для преобразования MMLU данных из формата CSV в JSONL"""

    def process_row(self, row) -> Dict[str, Any]:
        """
        Обрабатывает строку данных MMLU CSV и преобразует их в формат JSONL.
        Args:
            row: Строка данных из CSV файла
        Returns:
            dict: Запись в формате JSONL
        """
        question = row["question"]
        subject = row["subject"]
        choices_list = self.preprocess_options(row["choices"])
        answer_idx = int(row["answer"])
        answer_letter = self.get_letter_by_index(answer_idx)

        options_dict, options_text = self.create_options_data(choices_list)

        return {
            "instruction": self.instruction,
            "inputs": {
                "text": question,
                "subject": subject,
                "options": options_text,
                **options_dict,
            },
            "output": f"({answer_letter})",
            "meta": {"domain": subject},
        }

    def convert(self) -> bool:
        """
        Преобразовать данные MMLU из формата CSV в JSONL.
        Returns:
            bool: True, если преобразование прошло успешно, иначе False
        """
        if not self.validate_input():
            print(f"Входной файл не существует: {self.input_path}")
            return False

        try:
            df = pd.read_csv(self.input_path)
            ru_mmlu_data = []
            for _, row in df.iterrows():
                record = self.process_row(row)
                ru_mmlu_data.append(record)

            success = self.write_jsonl(ru_mmlu_data)
            if success:
                print(
                    f"Преобразовано {len(ru_mmlu_data)} записей из CSV. Результат сохранен в {self.output_path}"
                )
            return success
        except Exception as e:
            print(f"Ошибка при преобразовании CSV: {e}")
            return False


class MmluProCsvToJsonlConverter(CsvToJsonlConverter):
    """Конвертер для преобразования MMLU PRO данных из формата CSV в JSONL"""

    def process_row(self, row) -> Dict[str, Any]:
        """
        Обрабатывает строку данных MMLU PRO CSV и преобразует их в формат JSONL.
        Args:
            row: Строка данных из CSV файла
        Returns:
            dict: Запись в формате JSONL
        """
        question = row["question"]
        options_list = self.preprocess_options(row["options"])

        if "answer_index" in row and not pd.isna(row["answer_index"]):
            answer_idx = int(row["answer_index"])
            answer_letter = self.get_letter_by_index(answer_idx)
        else:
            answer_letter = row["answer"]
            answer_idx = ord(answer_letter) - ord("A")

        options_dict, options_text = self.create_options_data(options_list)

        return {
            "instruction": self.instruction,
            "inputs": {
                "text": question,
                "subject": row["category"],
                "options": options_text,
                **options_dict,
            },
            "output": f"({answer_letter})",
            "meta": {
                "id": int(row["question_id"]),
                "domain": row["src"].replace("ori_mmlu-", ""),
            },
        }

    def convert(self) -> bool:
        """
        Преобразовать данные MMLU PRO из формата CSV в JSONL.
        Returns:
            bool: True, если преобразование прошло успешно, иначе False
        """
        if not self.validate_input():
            print(f"Входной файл не существует: {self.input_path}")
            return False

        try:
            df = pd.read_csv(self.input_path)
            ru_mmlu_data = []

            for _, row in df.iterrows():
                record = self.process_row(row)
                ru_mmlu_data.append(record)

            success = self.write_jsonl(ru_mmlu_data)
            if success:
                print(
                    f"Преобразовано {len(ru_mmlu_data)} записей из CSV. Результат сохранен в {self.output_path}"
                )
            return success

        except Exception as e:
            print(f"Ошибка при преобразовании CSV: {e}")
            return False
