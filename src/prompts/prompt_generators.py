from .prompt_strategies import PromptStrategy
from abc import ABCMeta, abstractmethod

import json


class PromptGenerator(metaclass=ABCMeta):
    """Абстрактный класс для генерации промптов различных типов."""

    def __init__(self):
        self.data = []
        self.current_index = 0

    @abstractmethod
    def generate_prompt(self, item):
        pass

    def parse_jsonl(self, file_path):
        prompts = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line)
                prompts.append(json_obj)
        return prompts

    def load_data(self, file_path):
        self.data = self.parse_jsonl(file_path)
        self.current_index = 0
        return self

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
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
        super().__init__()
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def generate_prompt(self, item):
        instruction = item.get("instruction")
        inputs = item.get("inputs", {})

        return self.strategy.process(instruction, inputs)


class FewShotPromptGenerator(PromptGenerator):
    def __init__(self, strategy: PromptStrategy, n_shots: int = 5):
        super().__init__()
        if n_shots < 1:
            raise ValueError("Количество few-shot примеров (n_shots) должно быть не менее 1")
        if strategy is None:
            raise ValueError("Необходимо предоставить стратегию форматирования промпта")
        self.strategy = strategy
        self.n_shots = n_shots
        self.few_shot_examples = []

    def load_data(self, file_path):
        super().load_data(file_path)
        if len(self.data) <= self.n_shots:
            raise ValueError(
                f"Недостаточно данных ({len(self.data)}) для создания {self.n_shots} few-shot примеров."
            )
        
        self.few_shot_examples = self.data[:self.n_shots]
        self.data = self.data[self.n_shots:]
        self.current_index = 0
        return self

    def _format_example(self, item, include_answer: bool) -> str:
        """Форматирует один пример с использованием заданной стратегии."""
        instruction = item.get("instruction", "")
        inputs = item.get("inputs", {})
        answer = item.get("output", "")

        formatted_prompt = self.strategy.process(instruction, inputs)

        if include_answer:
            return f"<client>\n{formatted_prompt}\n<client>\n<model>\n({answer})\n<model>"
        else:
            return f"<client>\n{formatted_prompt}\n<client>\n<model>"

    def generate_prompt(self, item):
        """Генерирует полный few-shot промпт."""
        few_shot_prompts = [
            self._format_example(example, include_answer=True)
            for example in self.few_shot_examples
        ]

        current_prompt = self._format_example(item, include_answer=False)
        full_prompt = "\n\n".join(few_shot_prompts + [current_prompt])
        return full_prompt
