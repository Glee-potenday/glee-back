import pytest
from bson import ObjectId

from app.core.enums import SuggestionTagType, ContentLength
from app.suggester.suggester_document import SuggesterDocument
from app.suggester.suggester_service import SuggesterService
from datetime import datetime

from app.user.user_document import UserDocument


@pytest.mark.asyncio
async def test_create_suggestion() -> None:
    """ai 추천 데이터를 저장하는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY, SuggestionTagType.COMFORT]
    suggestion = "Test ai generated suggestion"
    title = "Test title"

    document = await SuggesterService.create_suggestion(user_id, title, suggestion, tag)

    assert document.user_id == user_id
    assert document.suggestion == suggestion
    assert isinstance(document.created_at, datetime)


@pytest.mark.asyncio
async def test_get_suggestion_by_id() -> None:
    """저장된 ai 추천 데이터를 가져오는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY, SuggestionTagType.COMFORT]
    suggestion = "Fetching from DB"
    title = "Test title"
    document = await SuggesterService.create_suggestion(user_id, title, suggestion, tag)

    retrieved_document = await SuggesterService.get_suggestion_by_id(str(document.id))

    assert retrieved_document is not None
    assert retrieved_document.id == document.id
    assert retrieved_document.suggestion == suggestion


@pytest.mark.asyncio
async def test_delete_suggestion() -> None:
    """저장된 ai 추천 데이터를 삭제하는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY, SuggestionTagType.COMFORT]
    suggestion = "This will be deleted"
    title = "Test title"
    document = await SuggesterService.create_suggestion(user_id, title, suggestion, tag)
    success = await SuggesterService.delete_suggestion(str(document.id))
    assert success is True

    # 삭제 후 다시 조회
    with pytest.raises(Exception):
        await SuggesterService.get_suggestion_by_id(str(document.id))


@pytest.mark.asyncio
async def test_update_suggestion_tags() -> None:
    """저장된 ai 추천 데이터를 삭제하는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY, SuggestionTagType.COMFORT]
    update_tag = [SuggestionTagType.SCHOOL, SuggestionTagType.COMFORT]
    suggestion = "update suggestion"
    title = "Test title"
    document = await SuggesterService.create_suggestion(user_id, title, suggestion, tag)

    updated_document = await SuggesterService.update_suggestion_tags(str(document.id), update_tag)

    assert updated_document.id == document.id
    assert updated_document.tag == update_tag


@pytest.mark.asyncio
async def test_update_suggestion() -> None:
    """저장된 ai 추천 데이터를 삭제하는 서비스 로직 테스트"""
    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY, SuggestionTagType.COMFORT]
    suggestion = "update suggestion"
    update_suggestion = "update suggestion"
    title = "Test title"
    document = await SuggesterService.create_suggestion(user_id, title, suggestion, tag)
    updated_document = await SuggesterService.update_suggestion(str(document.id), title, update_suggestion, tag)

    assert updated_document.id == document.id
    assert updated_document.title == title
    assert updated_document.suggestion == update_suggestion


@pytest.mark.asyncio
async def test_get_recommend_suggestions(exists_suggestion: SuggesterDocument) -> None:
    """추천 데이터를 가져오는 서비스 로직 테스트"""
    query = exists_suggestion.suggestion[:1]
    recommend_documents = await SuggesterService.get_recommend_suggestions(query)

    assert len(recommend_documents) > 0
    assert recommend_documents[0].recommend is True


@pytest.mark.asyncio
async def test_search_suggestions(exists_suggestion: SuggesterDocument, test_user: UserDocument) -> None:
    """추천 데이터를 검색하는 서비스 로직 테스트"""

    suggestions = await SuggesterService.find_suggestions_by_text("Test", test_user.id)
    tags = [tag.value for tag in exists_suggestion.tag]
    assert len(suggestions) > 0
    assert "Test" in suggestions[0].suggestion
    assert suggestions[0].user_id == test_user.id
    assert str(suggestions[0].tag) == str(tags)


@pytest.mark.asyncio
async def test_count(exists_suggestion: SuggesterDocument, test_user: UserDocument) -> None:
    """추천 데이터를 검색하는 서비스 로직 테스트"""

    suggestions = await SuggesterService.find_suggestions_by_text("Test", test_user.id)
    tags = [tag.value for tag in exists_suggestion.tag]

    assert len(suggestions) > 0
    assert "Test" in suggestions[0].suggestion
    assert suggestions[0].user_id == test_user.id
    assert str(suggestions[0].tag) == str(tags)


@pytest.mark.asyncio
async def test_get_suggestion_counts(exists_suggestion: SuggesterDocument) -> None:
    """추천 데이터를 검색하는 서비스 로직 테스트"""

    user_id = ObjectId()
    tag = [SuggestionTagType.APOLOGY, SuggestionTagType.COMFORT]
    suggestion = "Test ai generated suggestion"
    title = "Test title"

    await SuggesterService.create_suggestion(user_id, title, suggestion, tag)
    await SuggesterService.create_suggestion(user_id, title, suggestion, tag)

    my_suggestion_count = await SuggesterService.get_user_suggestion_count(user_id)
    recommend_suggestion_count = await SuggesterService.get_recommend_suggestion_count()

    assert my_suggestion_count == 2
    assert recommend_suggestion_count == 1


@pytest.mark.asyncio
async def test_regenerate_suggestions() -> None:

    exist_suggestion = "내가 저번에 한 말 때문에 상처 받았다면 정말 미안해. 내가 너무 생각없이 말한 것 같아. 다신 이런일 없도록 조심할게"
    length = ContentLength.EXTEND.value
    detail = "앞으로 잘 지내자는 내용을 추가 해줘"
    response = await SuggesterService.regenerate_suggestions(exist_suggestion, length, detail)
    titles, suggestions = response.titles, response.suggestions
    assert len(suggestions) > 0
    assert len(titles) > 0
    assert len(suggestions[0]) > len(exist_suggestion)
