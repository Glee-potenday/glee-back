import asyncio
from asyncio import AbstractEventLoop
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest_asyncio
from bson import ObjectId

from app.history.history_collection import HistoryCollection
from app.history.history_document import HistoryDocument, HistoryDTO
from app.suggester.suggester_document import SuggesterDocument, SuggesterDTO
from app.user.user_document import UserDocument
from app.user.user_dto import UserData
from app.user.user_service import UserService
from app.utils import mongo
from app.utils.jwt_handler import JwtHandler
from app.utils.jwt_payload import JwtPayload
from app.utils.models.suggestion import Suggestion
from app.core.enums import SuggestionTagType


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_db() -> None:
    for collection_name in await mongo.db.list_collection_names():
        await mongo.db[collection_name].drop()


@pytest_asyncio.fixture(scope="function")
async def test_user() -> UserDocument:
    kakao_id = 1
    nickname = "개발 계정"
    profile_image = ""
    thumbnail_image = ""

    user_data = UserData(
        kakao_id=kakao_id, nickname=nickname, profile_image=profile_image, thumbnail_image=thumbnail_image
    )

    user_id = await UserService.create_or_update_user(user_data)

    return UserDocument(
        _id=ObjectId(user_id),
        kakao_id=kakao_id,
        nickname=nickname,
        profile_image=profile_image,
        thumbnail_image=thumbnail_image,
    )


@pytest_asyncio.fixture(scope="function", autouse=True)
async def test_user_token(test_user: UserDocument) -> str:
    payload = JwtPayload(id=int(test_user.kakao_id), nickname=test_user.nickname)
    refresh_jwt = JwtHandler.create_jwt_token(payload.model_dump())

    return refresh_jwt


@pytest_asyncio.fixture(scope="function")
async def auth_header(test_user_token: str) -> dict[str, str]:
    """테스트용 JWT 인증 헤더 생성"""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest_asyncio.fixture(scope="function")
async def exists_suggestion(test_user: UserDocument) -> SuggesterDocument:
    suggestion_collection = mongo.db["suggester"]

    suggester_dto = SuggesterDTO(
        title="Test title",
        user_id=ObjectId(test_user.id),
        tag=[SuggestionTagType.SCHOOL.value, SuggestionTagType.APOLOGY.value],
        suggestion="Test suggestion",
        updated_at=datetime.now(),
        created_at=datetime.now(),
        recommend=True,
    )

    result = await suggestion_collection.insert_one(asdict(suggester_dto))
    tag = [SuggestionTagType(tag) for tag in suggester_dto.tag]
    return SuggesterDocument(
        _id=result.inserted_id,  # ✅ 삽입된 ObjectId 사용
        user_id=suggester_dto.user_id,
        title=suggester_dto.title,
        tag=tag,
        suggestion=suggester_dto.suggestion,
        created_at=suggester_dto.created_at,
        updated_at=suggester_dto.updated_at,
    )


@pytest_asyncio.fixture(scope="function")
async def exists_history(test_user: UserDocument) -> HistoryDocument:
    history_dto = HistoryDTO(
        user_id=test_user.id,
        suggestions=[
            Suggestion(title="First test", content="First test"),
            Suggestion(title="Second test", content="Second test"),
            Suggestion(title="Third test", content="Third test"),
        ],
        updated_at=datetime.now(),
        created_at=datetime.now(),
    )

    return await HistoryCollection.create(history_dto)


@pytest_asyncio.fixture(scope="function")
async def test_image_path() -> Path:
    return Path(__file__).parent / "assets/test.png"
