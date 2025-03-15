import os
import json
import yaml
import httpx
import asyncio
import random
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from httpx import ConnectTimeout, ReadTimeout

from app.core.settings import settings
from ai.utils.get_headers_payloads import get_headers_payloads
from ai.utils.deduplicate_sentence import deduplicate_sentences


class ReplySuggestion:
    def __init__(self) -> None:
        """답변 추천을 위한 클래스"""
        self.BASE_URL: str = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"
        self.BEARER_TOKEN: Optional[str] = os.getenv("CLOVA_AI_BEARER_TOKEN") or settings.CLOVA_AI_BEARER_TOKEN
        self.BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

        # 대체 답변 목록
        self.fallback_replies: list[str] = [
            "죄송합니다만, 현재 서비스 연결에 문제가 있어 답변을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요.",
            "네트워크 연결 문제로 인해 답변을 생성하지 못했습니다. 다시 시도해 주시겠어요?",
            "서비스 연결이 원활하지 않습니다. 잠시 후에 다시 시도해 주세요.",
            "일시적인 서버 연결 문제가 발생했습니다. 곧 해결될 예정이니 잠시 후 다시 시도해 주세요.",
            "현재 서비스가 혼잡하여 응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
        ]

    def _load_config(self, config_name: str) -> dict[str, Any]:
        """YAML 설정 파일을 로드하는 함수"""
        config_path: Path = self.BASE_DIR / "config" / config_name
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as file:
            config: dict[str, Any] = yaml.safe_load(file)
            return config

    async def _fetch_reply(self, client: httpx.AsyncClient, input_text: str, config_name: str) -> str:
        """비동기적으로 AI API 요청을 보내고 응답을 처리"""
        headers, payload = get_headers_payloads(
            str(self.BASE_DIR / "config" / config_name), input_text, random_seed=True
        )

        try:
            response: httpx.Response = await client.post(self.BASE_URL, headers=headers, json=payload, timeout=15.0)

            if response.status_code == 200:
                return await self._process_stream_response(response)
            else:
                logger.error(f"API 응답 오류: {response.status_code} - {response.text}")
                return self._get_fallback_reply(input_text)

        except (ConnectTimeout, ReadTimeout) as e:
            logger.error(f"API 연결 시간 초과: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 오류: {e}")
        except Exception as e:
            logger.error(f"API 요청 중 오류 발생: {e}")

        return self._get_fallback_reply(input_text)

    def _get_fallback_reply(self, input_text: str) -> str:
        """API 연결 실패 시 대체 답변 반환"""
        fallback_reply: str = random.choice(self.fallback_replies)
        logger.info(f"대체 답변 사용: {fallback_reply}")
        return fallback_reply

    async def generate_suggestions(self, input_text: str, config_name: str, num_suggestions: int = 3) -> list[str]:
        """비동기로 여러 개의 답변을 생성"""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                tasks = [self._fetch_reply(client, input_text, config_name) for _ in range(num_suggestions)]
                suggestions: list[str | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

            processed_suggestions: list[str] = []
            for suggestion in suggestions:
                if isinstance(suggestion, BaseException):
                    logger.error(f"답변 생성 중 오류 발생: {suggestion}")
                    processed_suggestions.append(self._get_fallback_reply(input_text))
                else:
                    processed_suggestions.append(suggestion)

            unique_suggestions: list[str] = list(dict.fromkeys(processed_suggestions))

            while len(unique_suggestions) < num_suggestions:
                fallback: str = self._get_fallback_reply(input_text)
                if fallback not in unique_suggestions:
                    unique_suggestions.append(fallback)

            for suggestion in unique_suggestions:
                logger.info(f"생성된 답변: {suggestion}")

            return unique_suggestions[:num_suggestions]

        except Exception as e:
            logger.error(f"답변 생성 중 예상치 못한 오류 발생: {e}")
            return [self._get_fallback_reply(input_text) for _ in range(num_suggestions)]

    async def _process_stream_response(self, response: httpx.Response) -> str:
        """비동기적으로 스트림 응답을 처리하여 텍스트 추출"""
        reply_text: str = ""
        previous_token: str = ""

        try:
            async for line in response.aiter_lines():
                if line and line.startswith("data:"):
                    data_str: str = line[len("data:") :].strip()
                    try:
                        data_json: dict[str, Any] = json.loads(data_str)
                        token: str = data_json.get("message", {}).get("content", "")

                        if token != previous_token:
                            reply_text += token
                            previous_token = token
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 디코딩 오류 발생: {e}, 원본 데이터: {data_str}")
                        continue
                    except Exception as e:
                        logger.error(f"스트림 응답 처리 중 오류 발생: {e}")
                        continue

            if not reply_text.strip():
                logger.warning("서버 응답이 비어 있음.")
                return self._get_fallback_reply("빈 응답")

            return deduplicate_sentences(reply_text.strip())

        except Exception as e:
            logger.error(f"스트림 응답 처리 중 예상치 못한 오류 발생: {e}")
            return self._get_fallback_reply("응답 처리 오류")

    async def generate_basic_reply(self, situation_text: str) -> list[str]:
        """상황 -> 답변 생성 함수 (비동기)"""
        return await self.generate_suggestions(situation_text, "config_reply_suggestions.yaml")

    async def generate_detailed_reply(
        self,
        situation_text: str,
        accent: Optional[str] = None,
        purpose: Optional[str] = None,
        detailed_description: str = "없음",
    ) -> list[str]:
        """상황, 말투, 용도, 추가 설명을 포함한 답변 생성 함수 (비동기)"""
        input_text: str = f"상황: {situation_text}"
        if accent and purpose:
            input_text += f"\n말투: {accent}\n용도: {purpose}"
        if detailed_description != "없음":
            input_text += f"\n사용자가 추가적으로 제공하는 디테일한 내용: {detailed_description}"

        return await self.generate_suggestions(input_text, "config_reply_suggestions_accent_purpose.yaml")
