import pytest
from bson import ObjectId

from app.core.enums import SuggestionTagType
from app.suggester.suggester_service import SuggesterService
from datetime import datetime


@pytest.mark.asyncio
async def test_create_suggestion() -> None:
    """AI 추천 데이터를 저장하는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY.value, SuggestionTagType.COMFORT.value]
    suggestion = "Test AI generated suggestion"

    document = await SuggesterService.create_suggestion(user_id, tag, suggestion)

    assert document.user_id == user_id
    assert document.suggestion == suggestion
    assert isinstance(document.created_at, datetime)


@pytest.mark.asyncio
async def test_get_suggestion_by_id() -> None:
    """저장된 AI 추천 데이터를 가져오는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY.value, SuggestionTagType.COMFORT.value]
    suggestion = "Fetching from DB"

    document = await SuggesterService.create_suggestion(user_id, tag, suggestion)

    retrieved_document = await SuggesterService.get_suggestion_by_id(str(document.id))

    assert retrieved_document is not None
    assert retrieved_document.id == document.id
    assert retrieved_document.suggestion == suggestion


@pytest.mark.asyncio
async def test_delete_suggestion() -> None:
    """저장된 AI 추천 데이터를 삭제하는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY.value, SuggestionTagType.COMFORT.value]
    suggestion = "This will be deleted"

    document = await SuggesterService.create_suggestion(user_id, tag, suggestion)

    success = await SuggesterService.delete_suggestion(str(document.id))
    assert success is True

    # 삭제 후 다시 조회
    with pytest.raises(Exception):
        await SuggesterService.get_suggestion_by_id(str(document.id))
