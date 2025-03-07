class OcrAgent:
    """OCR 처리를 담당하는 에이전트"""

    def __init__(self, max_retries=2):
        self.max_retries = max_retries
        self.post_processor = OcrPostProcessingAgent()
        self.preprocessor = ImagePreprocessor()

    def run(self, image_files: List[Tuple[str, bytes]]) -> str:
        """이미지 파일에서 텍스트를 추출합니다."""
        aggregated_text = []
        temp_files = []  # 임시 파일 경로 저장 리스트

        try:
            for filename, filedata in image_files:
                # 이미지 전처리 적용
                processed_bytes = self.preprocessor.preprocess(filedata)

                # 캐시 확인
                file_hash = ocr_cache.get_hash(processed_bytes)
                cached_result = ocr_cache.get(file_hash)
                if cached_result:
                    logger.info(f"Using cached OCR result for {filename}")
                    aggregated_text.append(cached_result)
                    continue

                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                    temp_file.write(processed_bytes)
                    temp_file_path = temp_file.name
                    temp_files.append(temp_file_path)  # 임시 파일 경로 저장

                # OCR 처리
                retry = 0
                while retry <= self.max_retries:
                    # 파일 경로 리스트만 전달
                    ocr_result = ocr_service.CLOVA_OCR([temp_file_path])
                    if isinstance(ocr_result, str) and ocr_result.startswith("Error"):
                        logger.error(ocr_result)
                        aggregated_text.append("")
                        break
                    extracted_text = extract_text_from_ocr_result(ocr_result)
                    if len(extracted_text.strip()) < 5 and retry < self.max_retries:
                        retry += 1
                        continue
                    else:
                        aggregated_text.append(extracted_text)
                        ocr_cache.set(file_hash, extracted_text)
                        break

            raw_text = "\n".join(aggregated_text)
            processed_text = self.post_processor.run(raw_text)
            return processed_text

        finally:
            # 모든 임시 파일 삭제
            for temp_file_path in temp_files:
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.error(f"임시 파일 삭제 중 오류 발생: {e}")