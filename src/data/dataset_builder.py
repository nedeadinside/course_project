import os
from typing import Dict, Any, List, Type

from .converters import DataConverter


class DatasetBuilder:
    """
    Класс для сборки наборов данных, используя различные конвертеры.
    Позволяет создавать, объединять и обрабатывать наборы данных.
    """

    def __init__(self, output_dir: str, default_instruction: str = None):
        """
        Инициализация построителя наборов данных.

        Args:
            output_dir: Директория для сохранения результатов
            default_instruction: Инструкция по умолчанию для задачи
        """
        self.output_dir = output_dir
        self.default_instruction = default_instruction
        self.converters = {}
        self.datasets = {}

        os.makedirs(output_dir, exist_ok=True)

    def register_converter(self, name: str, converter_class: Type[DataConverter]):
        """
        Регистрирует новый класс конвертера.

        Args:
            name: Имя конвертера
            converter_class: Класс конвертера
        """
        self.converters[name] = converter_class

    def add_dataset(
        self,
        name: str,
        input_path: str,
        converter_name: str,
        output_filename: str = None,
        instruction: str = None,
        converter_params: Dict[str, Any] = None,
    ):
        """
        Добавляет новый набор данных для обработки.

        Args:
            name: Имя набора данных
            input_path: Путь к входным данным
            converter_name: Имя зарегистрированного конвертера
            output_filename: Имя выходного файла (по умолчанию = name.jsonl)
            instruction: Инструкция для задачи (переопределяет default_instruction)
            converter_params: Дополнительные параметры для конвертера
        """
        if converter_name not in self.converters:
            raise ValueError(f"Конвертер '{converter_name}' не зарегистрирован")

        if not output_filename:
            output_filename = f"{name}.jsonl"

        output_path = os.path.join(self.output_dir, output_filename)

        dataset_info = {
            "name": name,
            "input_path": input_path,
            "output_path": output_path,
            "converter_name": converter_name,
            "instruction": instruction or self.default_instruction,
            "converter_params": converter_params or {},
        }

        self.datasets[name] = dataset_info

    def build_dataset(self, name: str) -> bool:
        """
        Строит указанный набор данных.

        Args:
            name: Имя набора данных для сборки

        Returns:
            bool: True если сборка успешна, иначе False
        """
        if name not in self.datasets:
            raise ValueError(f"Набор данных '{name}' не найден")

        dataset_info = self.datasets[name]
        converter_class = self.converters[dataset_info["converter_name"]]

        converter = converter_class(
            input_path=dataset_info["input_path"],
            output_path=dataset_info["output_path"],
            instruction=dataset_info["instruction"],
            **dataset_info["converter_params"],
        )

        success = converter.convert()

        if success:
            print(
                f"Набор данных '{name}' успешно собран и сохранен в {dataset_info['output_path']}"
            )
        else:
            print(f"Ошибка при сборке набора данных '{name}'")

        return success

    def build_all_datasets(self) -> Dict[str, bool]:
        """
        Строит все зарегистрированные наборы данных.

        Returns:
            Dict[str, bool]: Словарь с результатами сборки для каждого набора данных
        """
        results = {}
        for name in self.datasets:
            results[name] = self.build_dataset(name)
        return results

    def get_registered_converters(self) -> List[str]:
        """
        Возвращает список зарегистрированных конвертеров.

        Returns:
            List[str]: Список имен зарегистрированных конвертеров
        """
        return list(self.converters.keys())

    def get_registered_datasets(self) -> List[str]:
        """
        Возвращает список зарегистрированных наборов данных.

        Returns:
            List[str]: Список имен зарегистрированных наборов данных
        """
        return list(self.datasets.keys())

    def get_dataset_stats(self, name: str = None) -> Dict[str, Any]:
        """
        Возвращает статистику по набору данных или всем наборам.

        Args:
            name: Имя набора данных (если None, то для всех наборов)

        Returns:
            Dict[str, Any]: Статистика по набору данных
        """
        stats = {}

        if name is not None:
            if name not in self.datasets:
                raise ValueError(f"Набор данных '{name}' не найден")
            datasets_to_check = {name: self.datasets[name]}
        else:
            datasets_to_check = self.datasets

        for dataset_name, dataset_info in datasets_to_check.items():
            output_path = dataset_info["output_path"]
            if os.path.exists(output_path):
                line_count = 0
                with open(output_path, "r", encoding="utf-8") as f:
                    for _ in f:
                        line_count += 1

                stats[dataset_name] = {
                    "path": output_path,
                    "size": os.path.getsize(output_path),
                    "records": line_count,
                    "exists": True,
                }
            else:
                stats[dataset_name] = {"path": output_path, "exists": False}

        return stats
