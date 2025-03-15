import asyncio

from typing import List, Tuple

from loguru import logger


from ai.services.agent.image_pre_processor import ImagePreprocessor
from ai.services.agent.ocr_post_processing_agent import OcrPostProcessingAgent
from ai.utils.image_dto import ImageDto
from ai.utils.services import ocr_service


class OcrAgent:
    """ocr 처리를 담당하는 에이전트"""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self.post_processor: OcrPostProcessingAgent = OcrPostProcessingAgent()
        self.preprocessor: ImagePreprocessor = ImagePreprocessor()

    # 헬퍼 함수: ocr 결과 JSON에서 텍스트 추출
    async def extract_text_from_ocr_result(self, ocr_result: str) -> str:
        """ocr 결과에서 텍스트를 추출"""
        if isinstance(ocr_result, str):
            return ocr_result

        try:
            if isinstance(ocr_result, dict) and "images" in ocr_result:
                extracted_text = " ".join(
                    field["inferText"]
                    for image in ocr_result["images"]
                    if "fields" in image
                    for field in image["fields"]
                    if "inferText" in field
                )
                return extracted_text.strip()
        except Exception as e:
            logger.error(f"ocr 결과 파싱 중 오류 발생: {e}")

        return ""

    async def run(self, images: List[Tuple[str, bytes]]) -> str:
        """이미지 파일에서 텍스트를 추출합니다."""
        aggregated_text = []

        processed_data: list[ImageDto] = []
        for filename, filedata in images:
            # 이미지 전처리 적용
            processed_data.append(ImageDto(name=filename, data=self.preprocessor.preprocess(filedata)))

        retry = 0
        while retry <= self.max_retries:
            try:
                # 비동기 ocr 요청 (ClovaOcr.run() 사용)
                ocr_result = await ocr_service.run(processed_data)

                if isinstance(ocr_result, str) and ocr_result.startswith("Error"):
                    logger.error(ocr_result)
                    aggregated_text.append("")
                    break

                extracted_text = await self.extract_text_from_ocr_result(ocr_result)

                if len(extracted_text.strip()) < 5 and retry < self.max_retries:
                    retry += 1
                    logger.warning(f"ocr 결과가 너무 짧음, 재시도 {retry}/{self.max_retries}")
                    await asyncio.sleep(1)
                    continue
                else:
                    aggregated_text.append(extracted_text)
                    break

            except Exception as e:
                logger.error(f"ocr 처리 중 오류 발생: {str(e)}")
                if retry >= self.max_retries:
                    aggregated_text.append("")
                    break
                retry += 1
                await asyncio.sleep(1)

        raw_text = "\n".join(aggregated_text)
        processed_text = self.post_processor.run(raw_text)
        return processed_text
