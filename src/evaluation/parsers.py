from abc import ABCMeta, abstractmethod
import re


class ResponseParser(metaclass=ABCMeta):
    @abstractmethod
    def parse(self, response: str) -> str:
        pass


class MultipleChoiceParser(ResponseParser):
    def __init__(
        self, case_sensitive: bool = False, allowed_options: str = "ABCDEFGHIJ"
    ):
        self.case_sensitive = case_sensitive
        self.allowed_options = (
            allowed_options if case_sensitive else allowed_options.upper()
        )

    def parse(self, response: str) -> str:
        if not response:
            return ""

        patterns = [
            rf"(?:^|\s)(?:\(?([{re.escape(self.allowed_options)}])\)?)(?:[\.\s]|$)",
            rf"(?:ответ|answer)[:\s]*(?:\(?([{re.escape(self.allowed_options)}])\)?)(?:[\.\s]|$)",
            rf"(?:^|\s)(?:the answer is|мой ответ)[:\s]*(?:\(?([{re.escape(self.allowed_options)}])\)?)(?:[\.\s]|$)",
            rf"\(([{re.escape(self.allowed_options)}])\)",
            rf"([{re.escape(self.allowed_options)}])\.(?:\s|$)",
            rf"(?:^|\s)([{re.escape(self.allowed_options)}])$",
            rf"(?:вариант|option)[:\s]*([{re.escape(self.allowed_options)}])(?:[\.\s]|$)",
        ]

        for pattern in patterns:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            match = re.search(pattern, response, flags)
            if match:
                result = match.group(1)
                return result if self.case_sensitive else result.upper()

        return ""


class RegexParser(ResponseParser):
    def __init__(self, pattern: str, group: int = 1, fallback_to_full: bool = True):
        self.pattern = pattern
        self.group = group
        self.fallback_to_full = fallback_to_full

    def parse(self, response: str) -> str:
        if not response:
            return ""

        match = re.search(self.pattern, response)
        if match and self.group <= len(match.groups()):
            return match.group(self.group).strip()

        return response.strip() if self.fallback_to_full else ""
