from abc import ABCMeta, abstractmethod
import re


class ResponseParser(metaclass=ABCMeta):
    """
    Абстрактный базовый класс для парсеров ответов модели.
    """

    @abstractmethod
    def parse(self, response: str) -> str:
        """
        Извлекает ответ из выходных данных модели.

        Args:
            response (str): Полный ответ модели.

        Returns:
            str: Извлеченный ответ в нормализованной форме.
        """
        pass


class MultipleChoiceParser(ResponseParser):
    """
    Парсер для извлечения буквы ответа из ответа модели на задание с выбором вариантов.
    """

    def __init__(
        self, case_sensitive: bool = False, allowed_options: str = "ABCDEFGHIJ"
    ):
        """
        Инициализирует парсер для заданий с выбором вариантов.

        Args:
            case_sensitive (bool): Учитывать ли регистр при выделении буквы ответа.
            allowed_options (str): Строка с допустимыми буквами вариантов ответа.
        """
        self.case_sensitive = case_sensitive
        self.allowed_options = (
            allowed_options if case_sensitive else allowed_options.upper()
        )

    def parse(self, response: str) -> str:
        """
        Извлекает букву ответа из ответа модели.

        Args:
            response (str): Ответ модели.

        Returns:
            str: Извлеченная буква ответа (A, B, C, D и т.д.).
        """
        if not response:
            return ""

        patterns = [
            # Одиночная буква или буква в скобках
            rf"(?:^|\s)(?:\(?([{re.escape(self.allowed_options)}])\)?)(?:[\.\s]|$)",
            # После "ответ" или "answer"
            rf"(?:ответ|answer)[:\s]*(?:\(?([{re.escape(self.allowed_options)}])\)?)(?:[\.\s]|$)",
            # После "the answer is"
            rf"(?:^|\s)the answer is[:\s]*(?:\(?([{re.escape(self.allowed_options)}])\)?)(?:[\.\s]|$)",
        ]

        for pattern in patterns:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            match = re.search(pattern, response, flags)
            if match:
                result = match.group(1)
                return result if self.case_sensitive else result.upper()

        return ""


class RegexParser(ResponseParser):
    """
    Парсер для извлечения ответов с помощью пользовательского регулярного выражения.
    """

    def __init__(self, pattern: str, group: int = 1, fallback_to_full: bool = True):
        """
        Инициализирует парсер с пользовательским регулярным выражением.

        Args:
            pattern (str): Шаблон регулярного выражения.
            group (int): Номер группы для извлечения.
            fallback_to_full (bool): Возвращать ли полный ответ, если шаблон не найден.
        """
        self.pattern = pattern
        self.group = group
        self.fallback_to_full = fallback_to_full

    def parse(self, response: str) -> str:
        """
        Применяет заданное регулярное выражение к ответу модели и извлекает группу.
        Args:
            response (str): Ответ модели.
        Returns:
            str: Извлеченный ответ.
        """
        if not response:
            return ""

        match = re.search(self.pattern, response)
        if match and self.group <= len(match.groups()):
            return match.group(self.group).strip()

        return response.strip() if self.fallback_to_full else ""
