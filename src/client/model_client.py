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
        Инициализирует клиент для взаимодействия с моделью.
        Args:
            host (str): Хост, где запущен сервер с моделью.
            port (int): Порт, на котором слушает сервер.
            endpoint (str): Эндпоинт для запросов к модели.
        """
        self.base_url = f"http://{host}:{port}{endpoint}"
        self.headers = {
            "Content-Type": "application/json",
        }

    def send_request(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Отправляет запрос к модели и возвращает полученный ответ.
        Args:
            prompt (str): Промпт для модели.
            max_tokens (int): Максимальное количество токенов в ответе.
            temperature (float): Температура для семплирования.
            deterministic (bool): Флаг для детерминистической генерации.

        Returns:
            Dict[str, Any]: Ответ модели в формате словаря.
        """
        if deterministic:
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "deterministic": True,
            }
        else:
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
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
        self.determ_url = f"http://{host}:{port}/api/v1/generate-determ"

    def process_dataset(
        self,
        generator: PromptGenerator,
        max_tokens: int = 512,
        temperature: float = 0.7,
        deterministic: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Обрабатывает набор данных из генератора промптов.
        Args:
            generator (PromptGenerator): Генератор промптов.
            max_tokens (int): Максимальное количество токенов в ответе.
            temperature (float): Температура для семплирования.
            deterministic (bool): Флаг для детерминистической генерации.

        Returns:
            List[Dict[str, Any]]: Список ответов модели с дополнительными метаданными.
        """
        results = []
        current_batch = []
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

            if len(current_batch) >= self.batch_size:
                batch_results = self._process_batch(
                    current_batch, max_tokens, temperature, deterministic
                )
                results.extend(batch_results)
                current_batch = []
                batch_index += 1
                print(f"Обработан пакет {batch_index}")

        if current_batch:
            batch_results = self._process_batch(
                current_batch, max_tokens, temperature, deterministic
            )
            results.extend(batch_results)
            print(f"Обработан финальный пакет {batch_index + 1}")

        return results

    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        deterministic: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Обрабатывает один пакет запросов.
        Args:
            batch (List[Dict[str, Any]]): Пакет запросов для обработки.
            max_tokens (int): Максимальное количество токенов в ответе.
            temperature (float): Температура для семплирования.
            deterministic (bool): Флаг для детерминистической генерации.

        Returns:
            List[Dict[str, Any]]: Результаты обработки пакета.
        """
        results = []

        for item in batch:
            if deterministic and max_tokens <= 3:
                response = self._send_determ_request(item["prompt"])
            else:
                response = self.send_request(
                    item["prompt"], max_tokens, temperature, deterministic
                )

            result = {
                "index": item["index"],
                "prompt": item["prompt"],
                "domain": item.get("domain", ""),
                "expected_output": item.get("expected_output", ""),
                "model_output": response.get("output", ""),
                "error": response.get("error", None),
            }

            results.append(result)

        return results

    def _send_determ_request(self, prompt: str) -> Dict[str, Any]:
        """
        Отправляет запрос на эндпоинт для детерминистической генерации одной буквы.
        Args:
            prompt (str): Промпт для модели.
        Returns:
            Dict[str, Any]: Ответ модели в формате словаря.
        """
        data = {"prompt": prompt}

        try:
            response = requests.post(
                self.determ_url,
                headers=self.headers,
                data=json.dumps(data),
                timeout=120,
            )

            response.raise_for_status()
            result = response.json()

            if "letter" in result:
                return {"output": result["letter"]}
            else:
                return result

        except Exception as e:
            print(f"Ошибка при запросе к детерминистическому эндпоинту: {e}")
            return {"error": str(e)}
