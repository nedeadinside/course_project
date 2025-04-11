from abc import ABCMeta, abstractmethod
import json


class PromptGenerator(metaclass=ABCMeta):
    """Абстрактный класс для генерации промптов различных типов."""

    def __init__(self):
        self.data = []
        self.current_index = 0

    @abstractmethod
    def generate_prompt(self, item):
        """
        Генерирует промпт на основе входных данных.
        Args:
            item (dict): Данные для генерации промпта.
        Returns:
            str: Сгенерированный промпт.
        """
        pass

    def parse_jsonl(self, file_path):
        """
        Читает JSONL файл и возвращает список объектов JSON.
        Args:
            file_path (str): Путь к JSONL файлу.
        Returns:
            list: Список объектов JSON.
        """
        prompts = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line)
                prompts.append(json_obj)
        return prompts

    def load_data(self, file_path):
        """Загружает данные из JSONL файла для итерации."""
        self.data = self.parse_jsonl(file_path)
        self.current_index = 0
        return self

    def __iter__(self):
        """Возвращает себя как итератор."""
        self.current_index = 0
        return self

    def __next__(self):
        """Возвращает следующий промпт."""
        if self.current_index >= len(self.data):
            raise StopIteration

        item = self.data[self.current_index]
        prompt = self.generate_prompt(item)

        result = {
            "index": self.current_index,
            "prompt": prompt,
            "domain": item.get("meta").get("domain", ""),
            "output": item.get("output", ""),
        }

        self.current_index += 1
        return result


class SinglePromptGenerator(PromptGenerator):
    """Класс для генерации промптов с использованием взаимозаменяемых стратегий."""

    def __init__(self, strategy=None):
        """
        Инициализирует генератор промптов с заданной стратегией.
        Args:
            strategy : Стратегия обработки промптов.
        """
        super().__init__()
        self.strategy = strategy

    def set_strategy(self, strategy):
        """
        Устанавливает новую стратегию обработки промптов.

        Args:
            strategy (PromptStrategy): Новая стратегия обработки.
        """
        self.strategy = strategy

    def generate_prompt(self, item):
        """
        Генерирует промпт, используя выбранную стратегию.

        Args:
            item (dict): Словарь с ключами 'instruction' и 'inputs'.
        Returns:
            str: Сгенерированный промпт.
        """
        instruction = item.get("instruction")
        inputs = item.get("inputs", {})

        return self.strategy.process(instruction, inputs)
