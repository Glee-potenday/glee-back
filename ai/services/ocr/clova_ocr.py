import asyncio
import json
import os
import time
import uuid
from typing import Any

import httpx
from loguru import logger

from ai.utils.image_dto import ImageDto
from app.core.settings import settings


class ClovaOcr:
    def __init__(self) -> None:
        """Clova OCR API를 사용하는 클래스"""
        self.URL: str = os.getenv("CLOVA_OCR_URL", settings.CLOVA_OCR_URL)
        self.SECRET_KEY: str = os.getenv("CLOVA_OCR_SECRET_KEY", settings.CLOVA_OCR_SECRET_KEY)

        if not self.URL or not self.SECRET_KEY:
            logger.error("OCR API URL 또는 SECRET_KEY가 설정되지 않았습니다.")

        self.client: httpx.AsyncClient = httpx.AsyncClient()

    async def ocr_request(self, image_data: bytes, filename: str) -> str:
        """비동기 OCR 요청을 보내고 텍스트를 추출하여 반환"""
        file_ext: str = filename.split(".")[-1].lower()

        request_json: dict[str, Any] = {
            "images": [{"format": file_ext, "name": filename}],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(round(time.time() * 1000)),
        }

        payload: dict[str, bytes] = {"message": json.dumps(request_json).encode("UTF-8")}
        files: list[tuple[str, bytes]] = [("file", image_data)]

        headers: dict[str, str] = {
            "X-ocr-SECRET": self.SECRET_KEY,
        }

        try:
            response: httpx.Response = await self.client.post(
                self.URL,
                headers=headers,
                data=payload,  # ✅ JSON 데이터는 data로 보냄
                files=files,  # ✅ 파일은 multipart/form-data로 보냄
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return self.extract_text_from_result(result, filename)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text} for file {filename}")
        except Exception as e:
            logger.error(f"Unexpected Error: {str(e)} for file {filename}")

        return ""

    async def close(self) -> None:
        """클라이언트 세션을 닫음"""
        await self.client.aclose()

    @staticmethod
    def extract_text_from_result(result: dict[str, Any], filename: str) -> str:
        """OCR 결과에서 텍스트를 추출하여 반환"""
        if "images" not in result or not result["images"]:
            logger.error(f"[{filename}] OCR 결과에 'images' 키가 없습니다: {result}")
            return ""

        if "fields" not in result["images"][0]:
            logger.error(f"[{filename}] OCR 결과에 'fields' 키가 없습니다: {result}")
            return ""

        extracted_text: str = " ".join(field["inferText"] for field in result["images"][0]["fields"])
        logger.info(f"[{filename}] 추출된 텍스트: {extracted_text.strip()}")
        return extracted_text.strip()

    async def run(self, images: list[ImageDto]) -> str:
        """
        여러 이미지 파일을 받아 각 파일에 대해 OCR 요청을 보내고, 결과를 합쳐 반환합니다.
        """
        if not images:
            raise ValueError("No image files provided.")

        logger.info(f"총 {len(images)}개의 파일을 처리합니다.")

        tasks = [self.ocr_request(image.data, image.name) for image in images]
        results: list[str] = await asyncio.gather(*tasks)

        return "\n".join(results).strip()
