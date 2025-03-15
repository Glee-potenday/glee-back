import os
import yaml
import random
from typing import Optional, Union, Dict, Any, Tuple
from app.core.settings import settings


def load_config(file_path: str) -> Dict[str, Any]:
    """YAML 파일을 로드하여 딕셔너리 형태로 반환합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def get_headers_payloads(
    config_path: Union[str, Dict[str, Any]], conversation: Optional[str] = None, random_seed: bool = False
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """헤더와 페이로드를 반환하는 함수"""

    # config_path가 문자열이면 파일에서 로드, 딕셔너리면 그대로 사용
    config: Dict[str, Any] = load_config(config_path) if isinstance(config_path, str) else config_path

    # 환경 변수에서 토큰과 요청 ID 가져오기
    BEARER_TOKEN: Optional[str] = settings.CLOVA_AI_BEARER_TOKEN
    REQUEST_ID: Optional[str] = settings.CLOVA_REQ_ID_REPLY_SUMMARY

    if not BEARER_TOKEN or not REQUEST_ID:
        raise ValueError("CLOVA_AI_BEARER_TOKEN 또는 CLOVA_REQ_ID_REPLY_SUMMARY 환경 변수가 설정되지 않았습니다.")

    # 헤더 생성
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": REQUEST_ID,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    # 기본 시스템 메시지 설정
    messages: list[Dict[str, str]] = [{"role": "system", "content": config["SYSTEM_PROMPT"]}]

    if conversation:
        messages.append({"role": "user", "content": conversation})

    # 랜덤 시드 설정
    seed: int = random.randint(0, 10000) if random_seed else config["HYPER_PARAM"]["seed"]

    # 페이로드 생성
    payload: Dict[str, Any] = {
        "messages": messages,
        "topP": config["HYPER_PARAM"]["topP"],
        "topK": config["HYPER_PARAM"]["topK"],
        "maxTokens": config["HYPER_PARAM"]["maxTokens"],
        "temperature": config["HYPER_PARAM"]["temperature"],
        "repeatPenalty": config["HYPER_PARAM"]["repeatPenalty"],
        "stopBefore": config["HYPER_PARAM"]["stopBefore"],
        "includeAiFilters": config["HYPER_PARAM"]["includeAiFilters"],
        "seed": seed,
    }

    return headers, payload
