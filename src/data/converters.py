from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import string
import re
import json
import os


class DataConverter(metaclass=ABCMeta):
    def __init__(self, input_path: str, output_path: str, instruction: str = None):
        self.instruction = instruction
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def convert(self) -> bool:
        pass

    def validate_input(self) -> bool:
        return os.path.exists(self.input_path)


class CsvToJsonlConverter(DataConverter, metaclass=ABCMeta):
    def write_jsonl(self, data: List[Dict[str, Any]]) -> bool:
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"Ошибка при записи JSONL: {e}")
            return False

    @abstractmethod
    def process_row(self, row: pd.Series) -> Dict[str, Any]:
        pass

    def convert(self) -> bool:
        if not self.validate_input():
            print(f"Входной файл не существует: {self.input_path}")
            return False

        try:
            df = pd.read_csv(self.input_path)
            jsonl_data = []
            
            for _, row in df.iterrows():
                record = self.process_row(row)
                jsonl_data.append(record)

            success = self.write_jsonl(jsonl_data)
            if success:
                print(
                    f"Преобразовано {len(jsonl_data)} записей из CSV. Результат сохранен в {self.output_path}"
                )
            return success
        except Exception as e:
            print(f"Ошибка при преобразовании CSV: {e}")
            return False


class JsonlToJsonlConverter(DataConverter, metaclass=ABCMeta):
    def read_jsonl(self) -> List[Dict[str, Any]]:
        data = []
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        except Exception as e:
            print(f"Ошибка при чтении JSONL файла: {e}")
            return []
    
    def write_jsonl(self, data: List[Dict[str, Any]]) -> bool:
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"Ошибка при записи JSONL: {e}")
            return False
    
    @abstractmethod
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def convert(self) -> bool:
        if not self.validate_input():
            print(f"Входной файл не существует: {self.input_path}")
            return False
        
        try:
            input_data = self.read_jsonl()
            if not input_data:
                print(f"Нет данных для обработки в файле: {self.input_path}")
                return False
            
            output_data = []
            for item in input_data:
                processed_item = self.process_item(item)
                output_data.append(processed_item)
            
            success = self.write_jsonl(output_data)
            if success:
                print(
                    f"Преобразовано {len(output_data)} записей из JSONL. Результат сохранен в {self.output_path}"
                )
            return success
        except Exception as e:
            print(f"Ошибка при преобразовании JSONL: {e}")
            return False


class MultipleChoiceConverter:
    """Миксин для работы с вариантами выбора ответов"""
    
    @staticmethod
    def preprocess_options(options_str):
        clean_str = options_str.strip("[]").strip()

        pattern = r'(?:"([^"]*)")|(?:\'([^\']*)\')'
        matches = re.findall(pattern, clean_str)

        options = []
        for match in matches:
            option = match[0] if match[0] else match[1]
            if option.strip():
                options.append(option.strip())

        return options
    
    @staticmethod
    def get_letter_by_index(idx):
        return string.ascii_uppercase[idx]
    
    @staticmethod
    def create_options_data(options_list):
        options_dict = {}
        options_text = ""
        for i, option in enumerate(options_list):
            letter = MultipleChoiceConverter.get_letter_by_index(i)
            options_text += f"{letter} {option}\n"
            options_dict[f"option_{letter.lower()}"] = option

        return options_dict, options_text.strip()


class MmluCsvToJsonlConverter(CsvToJsonlConverter, MultipleChoiceConverter):
    def process_row(self, row) -> Dict[str, Any]:
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
            "output": f"{answer_letter}",
            "meta": {"domain": subject},
        }


class MmluProCsvToJsonlConverter(CsvToJsonlConverter, MultipleChoiceConverter):
    def process_row(self, row) -> Dict[str, Any]:
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
            "output": f"{answer_letter}",
            "meta": {
                "id": int(row["question_id"]),
                "domain": row["src"].replace("ori_mmlu-", ""),
            },
        }


class XLSumJsonlConverter(JsonlToJsonlConverter):
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        article_id = item.get("id", "")
        title = item.get("title", "")
        text = item.get("text", "")
        summary = item.get("summary", "")
        
        return {
            "instruction": self.instruction,
            "inputs": {
                "title": title,
                "text": text
            },
            "output": summary,
            "meta": {
                "id": article_id,
                "source": "XLSum"
            }
        }
