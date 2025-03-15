"""
GleeAgent 클래스의 메서드들을 실제 API와 객체를 사용하여 테스트하는 파일입니다.
"""

import pytest
import os
from pathlib import Path
import sys
import asyncio
import time
from typing import List, Tuple
from loguru import logger

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

from ai.glee_agent import GleeAgent

# """
# python3 -m pytest ai/tests/ -v: 더 자세한 로그를 보고싶을떄 (verbose)
# python3 -m pytest ai/tests/ -v -s: 로깅 정보를 함께 보고싶을 때
# 또는
# pytest tests/test_ai_services.py -v
# pytest tests/test_ai_services.py -v -s
# """


@pytest.fixture
def test_image_path() -> str:  # 테스트 이미지의 경로를 반환하는 픽스쳐
    return os.path.join(root_dir, "AI", "OCR_Test1.png")


@pytest.fixture  # test1. input: 테스트 이미지의 경로 -> output: [(테스트 이미지의 이름, 데이터)]
def test_image_files(test_image_path: str) -> List[Tuple[str, bytes]]:
    if not os.path.exists(test_image_path):
        pytest.skip(f"테스트 이미지 파일이 존재하지 않습니다: {test_image_path}")

    with open(test_image_path, "rb") as f:
        image_data = f.read()

    image_name = os.path.basename(test_image_path)
    logger.info("test1. 테스트 이미지 경로 -> 테스트 이미지 파일 테스트")
    return [(image_name, image_data)]


@pytest.mark.asyncio  # 이미지 -> 상황 분석 텍스트
async def test_analyze_situation(test_image_files: List[Tuple[str, bytes]]) -> None:
    logger.info(f"test2. 이미지 파일 경로 -> 상황 분석 텍스트 테스트")
    start_time = time.time()
    result = await GleeAgent.analyze_situation(test_image_files)
    end_time = time.time()

    logger.info(f"이미지 파일: {test_image_files[0][0]}")
    logger.info(f"분석 결과: {result}")
    logger.info(f"처리 시간: {end_time - start_time:.2f}초")

    # 결과 검증
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.asyncio  # 이미지 -> 상황, 말투, 용도 분석 텍스트
async def test_analyze_situation_accent_purpose(test_image_files: List[Tuple[str, bytes]]) -> None:
    logger.info(f"test3. 이미지 파일 경로 -> 상황, 말투, 용도 분석 텍스트 테스트")
    start_time = time.time()
    situation, accent, purpose = await GleeAgent.analyze_situation_accent_purpose(test_image_files)
    end_time = time.time()

    logger.info(f"\n===== 테스트: 실제 이미지로 상황, 말투, 용도 분석 =====")
    logger.info(f"이미지 파일: {test_image_files[0][0]}")
    logger.info(f"situation: {situation}")
    logger.info(f"accent: {accent}")
    logger.info(f"purpose: {purpose}")
    logger.info(f"처리 시간: {end_time - start_time:.2f}초")

    # 결과 검증
    assert isinstance(situation, str)
    assert isinstance(accent, str)
    assert isinstance(purpose, str)
    assert len(situation.strip()) > 0
    assert len(accent.strip()) > 0
    assert len(purpose.strip()) > 0


@pytest.mark.asyncio  # 상황을 입력받았을 때 -> 제목,답변 생성 테스트
async def test_generate_suggestions_situation() -> None:
    logger.info(f"test4. 상황을 입력받았을 때 -> 제목, 답변 생성 테스트")

    # 모의 상황
    test_situation = "면접에서 자기소개를 어떻게 하면 좋을지 알려주세요."

    start_time = time.time()
    titles, replies = await GleeAgent.generate_suggestions_situation(test_situation)
    end_time = time.time()

    logger.info(f"상황: {test_situation}")
    logger.info(f"title:\n")
    for i, title in enumerate(titles, 1):
        print(f"  {i}. {title}")

    logger.info(f"reply:\n")
    for i, reply in enumerate(replies, 1):
        print(f"  {i}. {reply[:100]}..." if len(reply) > 100 else f"  {i}. {reply}")

    logger.info(f"처리 시간: {end_time - start_time:.2f}초")

    # 결과 검증
    assert isinstance(titles, list)
    assert isinstance(replies, list)
    assert len(titles) > 0
    assert len(replies) > 0
    for title in titles:
        assert isinstance(title, str)
        assert len(title.strip()) > 0
    for reply in replies:
        assert isinstance(reply, str)
        assert len(reply.strip()) > 0


@pytest.mark.asyncio  # 상황, 말투, 용도 -> 답변 생성
async def test_generate_reply_suggestions_accent_purpose() -> None:
    logger.info(f"test5. 상황, 말투, 용도에 맞는 답변 제안 생성 테스트")

    # 모의 상황 정의
    test_situation = "면접에서 자기소개를 어떻게 하면 좋을지 알려주세요."
    test_accent = "친절하고 전문적인 말투"
    test_purpose = "카카오톡"

    start_time = time.time()
    titles, replies = await GleeAgent.generate_reply_suggestions_accent_purpose(
        test_situation, test_accent, test_purpose
    )
    end_time = time.time()

    logger.info(f"test_situation: {test_situation}")
    logger.info(f"test_accent: {test_accent}")
    logger.info(f"test_purpose: {test_purpose}")
    logger.info(f"title:\n")
    for i, title in enumerate(titles, 1):
        print(f"  {i}. {title}")
    logger.info(f"reply:\n")
    for i, reply in enumerate(replies, 1):
        print(f"  {i}. {reply[:100]}..." if len(reply) > 100 else f"  {i}. {reply}")
    logger.info(f"처리 시간: {end_time - start_time:.2f}초")

    # 결과 검증
    assert isinstance(titles, list)
    assert isinstance(replies, list)
    assert len(titles) > 0
    assert len(replies) > 0
    for title in titles:
        assert isinstance(title, str)
        assert len(title.strip()) > 0
    for reply in replies:
        assert isinstance(reply, str)
        assert len(reply.strip()) > 0


@pytest.mark.asyncio  # 상황, 말투, 용도, 상세 설명 -> 답변 생성
async def test_generate_reply_suggestions_detail() -> None:
    logger.info(f"test6. 상황, 말투, 용도, 상세 설명에 맞는 답변 제안 생성 테스트")
    # 테스트 데이터 정의
    test_situation = "면접에서 자기소개를 어떻게 하면 좋을지 알려주세요."
    test_accent = "친절하고 전문적인 말투"
    test_purpose = "면접 준비 조언"
    test_details = "IT 회사에 지원하는 신입 개발자입니다."

    start_time = time.time()
    titles, replies = await GleeAgent.generate_reply_suggestions_detail(
        test_situation, test_accent, test_purpose, test_details
    )
    end_time = time.time()

    logger.info(f"test_situation: {test_situation}")
    logger.info(f"test_accent: {test_accent}")
    logger.info(f"test_purpose: {test_purpose}")
    logger.info(f"test_details: {test_details}")

    logger.info(f"title:\n")
    for i, title in enumerate(titles, 1):
        print(f"  {i}. {title}")

    logger.info(f"reply:\n")
    for i, reply in enumerate(replies, 1):
        print(f"  {i}. {reply[:100]}..." if len(reply) > 100 else f"  {i}. {reply}")

    logger.info(f"처리 시간: {end_time - start_time:.2f}초")

    # 결과 검증
    assert isinstance(titles, list)
    assert isinstance(replies, list)
    assert len(titles) > 0
    assert len(replies) > 0
    for title in titles:
        assert isinstance(title, str)
        assert len(title.strip()) > 0
    for reply in replies:
        assert isinstance(reply, str)
        assert len(reply.strip()) > 0


@pytest.mark.asyncio
async def test_generate_reply_suggestions_detail_length() -> None:
    logger.info(f"test7. 실제 길이 조정 및 추가 설명을 포함한 답변 제안 생성 테스트")
    # 테스트 데이터 정의
    test_suggestion = "면접에서 자기소개를 어떻게 하면 좋을지 알려주세요."
    test_length = "짧게"
    test_add_description = "IT 회사에 지원하는 신입 개발자입니다."

    # 시작 시간 기록
    start_time = time.time()

    # 실제 메서드 호출
    titles, replies = await GleeAgent.generate_reply_suggestions_detail_length(
        test_suggestion, test_length, test_add_description
    )

    # 종료 시간 기록
    end_time = time.time()

    logger.info(f"제안: {test_suggestion}")
    logger.info(f"길이: {test_length}")
    logger.info(f"추가 설명: {test_add_description}")

    logger.info(f"생성된 제목:")
    for i, title in enumerate(titles, 1):
        print(f"  {i}. {title}")

    logger.info(f"생성된 답변:")
    for i, reply in enumerate(replies, 1):
        print(f"  {i}. {reply[:100]}..." if len(reply) > 100 else f"  {i}. {reply}")

    logger.info(f"처리 시간: {end_time - start_time:.2f}초")

    # 결과 검증
    assert isinstance(titles, list)
    assert isinstance(replies, list)
    assert len(titles) > 0
    assert len(replies) > 0
    for title in titles:
        assert isinstance(title, str)
        assert len(title.strip()) > 0
    for reply in replies:
        assert isinstance(reply, str)
        assert len(reply.strip()) > 0
