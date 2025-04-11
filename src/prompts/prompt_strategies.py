from abc import ABCMeta, abstractmethod


class PromptStrategy(metaclass=ABCMeta):
    """Абстрактный класс стратегии для обработки различных форматов промптов."""

    @abstractmethod
    def process(self, instruction, inputs):
        """
        Обрабатывает инструкцию и входные данные для создания промпта.

        Args:
            instruction (str): Инструкция для промпта.
            inputs (dict): Входные данные для заполнения промпта.

        Returns:
            str: Сгенерированный промпт.
        """
        pass


class OptionsPromptStrategy(PromptStrategy):
    """Стратегия для обработки промптов с вариантами ответов."""

    def process(self, instruction, inputs):
        """Формирует промпт с вариантами ответов."""
        inputs_copy = inputs.copy()

        options = []
        for letter in "abcdefghij":
            option_key = f"option_{letter}"
            if option_key in inputs_copy:
                options.append(f"{letter.upper()}. {inputs_copy[option_key]}")

        inputs_copy["options"] = "\n".join(options)
        return instruction.format(**inputs_copy)
