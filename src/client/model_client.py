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
        max_tokens: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.base_url = f"http://{host}:{port}{endpoint}"
        self.headers = {"Content-Type": "application/json"}
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def send_request(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
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
        max_tokens: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        super().__init__(host, port, endpoint, max_tokens, temperature, top_p)
        self.batch_size = batch_size

    def send_batch_request(
        self,
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        data = {
            "prompts": prompts,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        response = requests.post(
            self.base_url, headers=self.headers, data=json.dumps(data)
        )

        response.raise_for_status()
        results = response.json()

        return results

    def process_dataset(
        self,
        generator: PromptGenerator,
    ) -> List[Dict[str, Any]]:
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
                batch_results = self._process_batch(current_batch, batch_prompts)
                results.extend(batch_results)

                current_batch = []
                batch_prompts = []
                batch_index += 1
                print(f"Обработан пакет {batch_index}")
        if current_batch:
            batch_results = self._process_batch(current_batch, batch_prompts)
            results.extend(batch_results)
            print(f"Обработан финальный пакет {batch_index + 1}")

        return results

    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_prompts: List[str],
    ) -> List[Dict[str, Any]]:
        batch_responses = self.send_batch_request(batch_prompts)
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
