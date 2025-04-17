from abc import ABCMeta, abstractmethod


class PromptStrategy(metaclass=ABCMeta):
    """Абстрактный класс стратегии для обработки различных форматов промптов."""

    @abstractmethod
    def process(self, instruction, inputs):
        pass


class OptionsPromptStrategy(PromptStrategy):
    """Стратегия для обработки промптов с вариантами ответов."""

    def process(self, instruction, inputs):
        inputs_copy = inputs.copy()

        options = []
        for letter in "abcdefghij":
            option_key = f"option_{letter}"
            if option_key in inputs_copy:
                options.append(f"{letter.upper()}. {inputs_copy[option_key]}")

        inputs_copy["options"] = "\n".join(options)
        return instruction.format(**inputs_copy)


class GenerationPromptStrategy(PromptStrategy):
    """Стратегия для обработки промптов для генерации текста."""

    def process(self, instruction, inputs):
        formatted_instruction = instruction.format(**inputs)
        return formatted_instruction
