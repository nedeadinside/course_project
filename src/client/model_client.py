import json
import requests
from typing import Dict, Any, List
from ..prompts import PromptGenerator


class ModelClient:
    """
    Класс для формирования промптов и отправки запросов на локальный веб-сервер с моделью.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        endpoint: str = "/api/v1/generate",
    ):
        """
        Инициализация клиента для запросов к модели.
        Args:
            host (str): Хост, где запущен сервер с моделью.
            port (int): Порт, на котором слушает сервер.
            endpoint (str): Эндпоинт для запросов к модели.
        """
        self.base_url = f"http://{host}:{port}{endpoint}"
        self.headers = {"Content-Type": "application/json"}

    def send_request(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Отправляет один запрос к модели.
        Args:
            prompt (str): Промпт для модели.
            max_tokens (int): Максимальное количество токенов в ответе.

        Returns:
            Dict[str, Any]: Ответ модели.
        """
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "deterministic": True,
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(data),
                timeout=120,
            )

            response.raise_for_status()
            result_json = response.json()

            if "letter" in result_json:
                return {"output": result_json["letter"]}
            else:
                return result_json

        except Exception as e:
            print(e)
            return {"error": str(e)}


class BatchModelClient(ModelClient):
    """
    Класс для пакетной обработки запросов к модели.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        endpoint: str = "/api/v1/generate",
        batch_size: int = 10,
    ):
        """
        Инициализирует клиент для пакетной обработки запросов.
        Args:
            host (str): Хост, где запущен сервер с моделью.
            port (int): Порт, на котором слушает сервер.
            endpoint (str): Эндпоинт для запросов к модели.
            batch_size (int): Размер пакета для запросов.
        """
        super().__init__(host, port, endpoint)
        self.batch_size = batch_size

    def send_batch_request(
        self,
        prompts: List[str],
        max_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        """
        Отправляет батч запросов к модели.
        Args:
            prompts (List[str]): Список промптов для модели.
            max_tokens (int): Максимальное количество токенов в ответе.

        Returns:
            List[Dict[str, Any]]: Список ответов модели.
        """
        data = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(data)
            )

            response.raise_for_status()
            results = response.json()
            
            return results
        except Exception as e:
            print(f"Ошибка при отправке батча запросов: {e}")
            return [{"error": str(e)} for _ in range(len(prompts))]

    def process_dataset(
        self,
        generator: PromptGenerator,
        max_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        """
        Обрабатывает набор данных из генератора промптов.
        Args:
            generator (PromptGenerator): Генератор промптов.
            max_tokens (int): Максимальное количество токенов в ответе.

        Returns:
            List[Dict[str, Any]]: Список ответов модели с дополнительными метаданными.
        """
        results = []
        current_batch = []
        batch_prompts = []
        batch_index = 0

        for item in generator:
            prompt = item["prompt"]
            current_batch.append(
                {
                    "index": item["index"],
                    "prompt": prompt,
                    "domain": item.get("domain", ""),
                    "expected_output": item.get("output", ""),
                }
            )
            batch_prompts.append(prompt)

            if len(current_batch) >= self.batch_size:
                batch_results = self._process_batch(current_batch, batch_prompts, max_tokens)
                results.extend(batch_results)
                
                current_batch = []
                batch_prompts = []
                batch_index += 1
                print(f"Обработан пакет {batch_index}")
        if current_batch:
            batch_results = self._process_batch(current_batch, batch_prompts, max_tokens)
            results.extend(batch_results)
            print(f"Обработан финальный пакет {batch_index + 1}")

        return results

    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_prompts: List[str],
        max_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Обрабатывает один пакет запросов.
        Args:
            batch (List[Dict[str, Any]]): Пакет запросов для обработки.
            batch_prompts (List[str]): Список промптов для отправки на сервер.
            max_tokens (int): Максимальное количество токенов в ответе.

        Returns:
            List[Dict[str, Any]]: Результаты обработки пакета.
        """
        batch_responses = self.send_batch_request(batch_prompts, max_tokens)
        results = []

        for item, response in zip(batch, batch_responses):
            result = {
                "index": item["index"],
                "prompt": item["prompt"],
                "domain": item.get("domain", ""),
                "expected_output": item.get("expected_output", ""),
                "model_output": response.get("text", ""),
                "error": response.get("error", None),
            }
            results.append(result)

        return results
