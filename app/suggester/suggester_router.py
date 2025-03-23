from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query

from ai.glee_agent import GleeAgent
from app.history.history_service import HistoryService
from app.suggester.suggester_request import (
    GenerateSuggestionRequest,
    SuggestionRequest,
    UpdateSuggestionTagsRequest,
    RegenerateSuggestionRequest,
)
from app.suggester.suggester_response import (
    AnalyzeImagesConversationResponse,
    SuggestionResponse,
    DeleteSuggestionResponse,
    GenerateSuggestion,
    GenerateSuggestionsResponse,
    SuggestionsResponse,
    SearchSuggestionResponse,
    GetSuggestionCounts,
)
from app.core.enums import PurposeType
from app.suggester.suggester_service import SuggesterService
from app.user.user_document import UserDocument
from app.utils.jwt_handler import JwtHandler
from app.utils.models.suggestion import Suggestion
from loguru import logger

router = APIRouter(prefix="/suggester", tags=["suggester"])


@router.get("/count", response_model=GetSuggestionCounts, summary="내 제안 개수 및 추천 제안 개수 가져오기")
async def get_suggestion_counts(
    user: UserDocument | None = Depends(JwtHandler.get_optional_current_user),  # ✅ JWT 인증된 사용자
) -> GetSuggestionCounts:

    my_suggestion_count = 0
    if user:
        logger.info(f"Fetching suggestion counts for user: {user.id}")
        my_suggestion_count = await SuggesterService.get_user_suggestion_count(user.id)

    recommended_suggestion_count = await SuggesterService.get_recommend_suggestion_count()

    logger.info(f" My suggestions: {my_suggestion_count}, Recommended suggestions: {recommended_suggestion_count}")

    return GetSuggestionCounts(
        user_suggestion_count=my_suggestion_count,
        recommended_suggestion_count=recommended_suggestion_count,
    )


@router.get("/search", response_model=SearchSuggestionResponse, summary="특정 단어가 포함된 제안 검색")
async def search_suggestions(
    query: str = Query(..., description="검색할 단어"),
    user: UserDocument = Depends(JwtHandler.get_current_user),  # ✅ JWT 인증된 사용자
) -> SearchSuggestionResponse:
    """본문에 특정 단어가 포함된 제안을 검색하는 API"""
    logger.info(f"User {user.nickname} {user.id} is searching for suggestions containing '{query}'")
    suggestions = await SuggesterService.find_suggestions_by_text(query, user.id)
    if not suggestions:
        logger.error(f"No suggestions found for query '{query}'")
        raise HTTPException(status_code=404, detail="No suggestions found")
    response = [
        SuggestionResponse(
            id=str(suggestion.id),
            title=suggestion.title,
            tags=suggestion.tag,
            suggestion=suggestion.suggestion,
            updated_at=suggestion.updated_at,
            created_at=suggestion.created_at,
        )
        for suggestion in suggestions
    ]
    return SearchSuggestionResponse(suggestions=response)


@router.get("/recommend", response_model=SuggestionsResponse)
async def get_recommend_suggestions(
    query: str | None = Query(None, description="검색할 단어"),
) -> SuggestionsResponse:
    logger.info(f"Request recommended suggestions - query {query}")
    suggestions = await SuggesterService.get_recommend_suggestions(query)
    suggestion_responses = [
        SuggestionResponse(
            id=str(suggestion.id),
            title=suggestion.title,
            tags=suggestion.tag,
            suggestion=suggestion.suggestion,
            updated_at=suggestion.updated_at,
            created_at=suggestion.created_at,
        )
        for suggestion in suggestions
    ]
    logger.info("Returning recommended suggestions")
    return SuggestionsResponse(
        suggestions=suggestion_responses,
    )


@router.post(
    "/analyze/image",
    summary="최대 사진 4장까지 보내면 ai 상황을 분석하여 대답함",
    response_model=AnalyzeImagesConversationResponse,
)
async def analyze_images(
    purpose: PurposeType = Form(...),
    image_file_1: Optional[UploadFile] = File(None),
    image_file_2: Optional[UploadFile] = File(None),
    image_file_3: Optional[UploadFile] = File(None),
    image_file_4: Optional[UploadFile] = File(None),
) -> AnalyzeImagesConversationResponse:

    logger.info("Received image analysis request")
    image_files = [file for file in [image_file_1, image_file_2, image_file_3, image_file_4] if file is not None]
    if len(image_files) > 4:
        logger.error("User tried to upload more than 4 images")
        raise HTTPException(status_code=400, detail="You can only upload up to 4 images.")

    elif len(image_files) == 0:
        logger.error("User tried to upload 0 images")
        raise HTTPException(status_code=400, detail="You must upload at least one image.")

    files_data = [(file.filename, await file.read()) for file in image_files if file and file.filename]

    if purpose == PurposeType.PHOTO_RESPONSE:
        situation = await GleeAgent.analyze_situation(files_data)
        tone = ""
        usage = ""
    elif purpose == PurposeType.SIMILAR_VIBE_RESPONSE:
        situation, tone, usage = await GleeAgent.analyze_situation_accent_purpose(files_data)
    else:
        logger.error(f"Failed to analyze images - Invalid purpose: {purpose}")
        raise HTTPException(status_code=400, detail="Invalid purpose.")
    logger.info(f"Analyzed images - Situation: {situation}, Tone: {tone}, Usage: {usage}, Purpose: {purpose}")
    return AnalyzeImagesConversationResponse(situation=situation, tone=tone, usage=usage, purpose=purpose)


@router.post(
    "/generate",
    summary="상황, 말투, 용도, 상세 정보를 받아 ai 글을 생성하여 반환",
    response_model=GenerateSuggestionsResponse,
)
async def generate_suggestion(
    request: GenerateSuggestionRequest,
    user: UserDocument | None = Depends(JwtHandler.get_optional_current_user),  # ✅ JWT 인증된 사용자
) -> GenerateSuggestionsResponse:
    logger.info(f"Generating suggestions - User: {user.nickname if user else 'Guest'}, Request: {request}")

    response = await SuggesterService.generate_suggestions(
        situation=request.situation, tone=request.tone, usage=request.usage, detail=request.detail
    )

    result = [
        GenerateSuggestion(title=title, content=suggestion)
        for title, suggestion in zip(response.titles, response.suggestions)
    ]

    logger.info(f"Generated suggestions - User: {user.nickname if user else 'Guest'}, Suggestions: {result}")

    if user:
        _suggestions = [Suggestion(title=suggestion.title, content=suggestion.content) for suggestion in result]
        await HistoryService.create_history(user.id, _suggestions)

    return GenerateSuggestionsResponse(suggestions=result)


@router.post(
    "/regenerate",
    summary="기존 생성된 글과 추가 정보를 받아 ai 글을 다시 생성하여 반환",
    response_model=GenerateSuggestionsResponse,
)
async def regenerate_suggestion(
    request: RegenerateSuggestionRequest,
    user: UserDocument | None = Depends(JwtHandler.get_optional_current_user),  # ✅ JWT 인증된 사용자
) -> GenerateSuggestionsResponse:
    logger.info(f"Regenerating suggestions - User: {user.nickname if user else 'Guest'}, Request: {request}")

    response = await SuggesterService.regenerate_suggestions(
        exist_suggestion=request.exist_suggestion, length=request.length.value, detail=request.detail
    )

    result = [
        GenerateSuggestion(title=title, content=suggestion)
        for title, suggestion in zip(response.titles, response.suggestions)
    ]

    logger.info(f"Regenerated suggestions - User: {user.nickname if user else 'Guest'}, Suggestions: {result}")

    if user:
        _suggestions = [Suggestion(title=suggestion.title, content=suggestion.content) for suggestion in result]
        await HistoryService.create_history(user.id, _suggestions)

    return GenerateSuggestionsResponse(suggestions=result)


@router.post("", response_model=SuggestionResponse, summary="유저가 생성한 글제안 - 저장")
async def save_suggestion(
    request: SuggestionRequest,
    user: UserDocument = Depends(JwtHandler.get_current_user),  # ✅ JWT 인증된 사용자
) -> SuggestionResponse:
    logger.info(f"User {user.id} is saving a suggestion: {request.title}")
    new_suggestion = await SuggesterService.create_suggestion(user.id, request.title, request.suggestion, request.tags)
    logger.info(f"Suggestion saved successfully with ID: {new_suggestion.id}")

    return SuggestionResponse(
        id=str(new_suggestion.id),
        title=new_suggestion.title,
        tags=new_suggestion.tag,
        suggestion=new_suggestion.suggestion,
        updated_at=new_suggestion.updated_at,
        created_at=new_suggestion.created_at,
    )


@router.get("/{suggestion_id}", response_model=SuggestionResponse, summary="글 제안 가져오기")
async def get_suggestion(
    suggestion_id: str,
    user: UserDocument = Depends(JwtHandler.get_current_user),  # ✅ JWT 인증된 사용자
) -> SuggestionResponse:
    logger.info(f"Fetching suggestion with ID: {suggestion_id}")
    suggestion = await SuggesterService.get_suggestion_by_id(suggestion_id)

    if not suggestion:
        logger.error(f"Suggestion with ID {suggestion_id} not found")
        raise HTTPException(status_code=404, detail="Suggestion not found")

    # ✅ 사용자가 자신의 데이터만 조회할 수 있도록 제한
    if suggestion.user_id != user.id:
        logger.error(f"Unauthorized access attempt by user {user.id} for suggestion {suggestion_id}")

        raise HTTPException(status_code=403, detail="Access denied")
    logger.info(f"Suggestion {suggestion_id} fetched successfully")

    return SuggestionResponse(
        id=str(suggestion.id),
        title=suggestion.title,
        tags=suggestion.tag,
        suggestion=suggestion.suggestion,
        updated_at=suggestion.updated_at,
        created_at=suggestion.created_at,
    )


@router.get("/user/me", response_model=SuggestionsResponse, summary="내 글 제안 가져오기")
async def get_my_suggestions(
    user: UserDocument = Depends(JwtHandler.get_current_user),  # ✅ JWT 인증된 사용자
) -> SuggestionsResponse:
    logger.info(f"Fetching suggestions for user: {user.id}")

    my_suggestions = await SuggesterService.get_suggestions_by_user(user.id)
    suggestion_responses = [
        SuggestionResponse(
            id=str(my_suggestion.id),
            title=my_suggestion.title,
            tags=my_suggestion.tag,
            suggestion=my_suggestion.suggestion,
            updated_at=my_suggestion.updated_at,
            created_at=my_suggestion.created_at,
        )
        for my_suggestion in my_suggestions
    ]
    logger.info(f"Fetched {len(my_suggestions)} suggestions for user {user.id}")

    return SuggestionsResponse(
        suggestions=suggestion_responses,
    )


@router.get("/user/summary")
async def get_my_suggestions_summary(
    user: UserDocument = Depends(JwtHandler.get_current_user),  # ✅ JWT 인증된 사용자
) -> SuggestionsResponse:
    logger.info(f"Fetching summary for user: {user.id}")

    my_suggestions = await SuggesterService.get_suggestions_by_user(user.id)
    suggestion_responses = [
        SuggestionResponse(
            id=str(my_suggestion.id),
            title=my_suggestion.title,
            tags=[],
            suggestion=my_suggestion.suggestion,
            updated_at=my_suggestion.updated_at,
            created_at=my_suggestion.created_at,
        )
        for my_suggestion in my_suggestions
    ]
    logger.info(f"Fetched summary with {len(my_suggestions)} suggestions for user {user.id}")
    return SuggestionsResponse(
        suggestions=suggestion_responses,
    )


@router.put("/{suggestion_id}", response_model=SuggestionResponse, summary="내 글 제안 수정")
async def update_suggestion(
    request: SuggestionRequest,
    suggestion_id: str,
    user: UserDocument = Depends(JwtHandler.get_current_user),  # ✅ JWT 인증된 사용자
) -> SuggestionResponse:
    logger.info(f"User {user.id} requested to update suggestion {suggestion_id}")

    suggestion = await SuggesterService.get_suggestion_by_id(suggestion_id)

    if not suggestion:
        logger.error(f"Suggestion {suggestion_id} not found")

        raise HTTPException(status_code=404, detail="Suggestion not found")

    # ✅ 사용자가 자신의 데이터만 수정할 수 있도록 제한
    if suggestion.user_id != user.id:
        logger.error(f"Failed to update suggestion {suggestion_id}")
        raise HTTPException(status_code=403, detail="Access denied")

    updated_suggestion = await SuggesterService.update_suggestion(
        suggestion_id, request.title, request.suggestion, request.tags
    )

    if not updated_suggestion:
        logger.error(f"Failed to update suggestion {suggestion_id}")
        raise HTTPException(status_code=500, detail="Failed to delete suggestion")
    logger.info(f"Suggestion {suggestion_id} updated successfully")

    return SuggestionResponse(
        id=str(updated_suggestion.id),
        title=updated_suggestion.title,
        tags=updated_suggestion.tag,
        suggestion=updated_suggestion.suggestion,
        updated_at=updated_suggestion.updated_at,
        created_at=updated_suggestion.created_at,
    )


@router.delete("/{suggestion_id}", response_model=DeleteSuggestionResponse, summary="내 글 제안 삭제")
async def delete_suggestion(
    suggestion_id: str,
    user: UserDocument = Depends(JwtHandler.get_current_user),  # ✅ JWT 인증된 사용자
) -> DeleteSuggestionResponse:
    suggestion = await SuggesterService.get_suggestion_by_id(suggestion_id)
    logger.info(f"User {user.id} requested to delete suggestion {suggestion_id}")
    if not suggestion:
        logger.error(f"Suggestion {suggestion_id} not found")

        raise HTTPException(status_code=404, detail="Suggestion not found")

    # ✅ 사용자가 자신의 데이터만 삭제할 수 있도록 제한
    if suggestion.user_id != user.id:
        logger.error(f"Unauthorized delete attempt by user {user.id} on suggestion {suggestion_id}")

        raise HTTPException(status_code=403, detail="Access denied")

    success = await SuggesterService.delete_suggestion(suggestion_id)
    if not success:
        logger.error(f"Failed to delete suggestion {suggestion_id}")

        raise HTTPException(status_code=500, detail="Failed to delete suggestion")

    return DeleteSuggestionResponse(
        message="Suggestion deleted successfully",
        deleted_suggestion_id=suggestion_id,
    )


@router.put("/tag")
async def update_suggestion_tag(
    request: UpdateSuggestionTagsRequest, user: UserDocument = Depends(JwtHandler.get_current_user)
) -> SuggestionResponse:
    logger.info(f"User {user.id} requested to update tags for suggestion {request.suggestion_id}")

    suggestion = await SuggesterService.get_suggestion_by_id(request.suggestion_id)

    if not suggestion:
        logger.error(f"Suggestion {request.suggestion_id} not found")

        raise HTTPException(status_code=404, detail="Suggestion not found")

    if suggestion.user_id != user.id:
        logger.error(f"Unauthorized tag update attempt by user {user.id} on suggestion {request.suggestion_id}")
        raise HTTPException(status_code=403, detail="Access denied")

    updated_suggestion = await SuggesterService.update_suggestion_tags(request.suggestion_id, request.tags)
    logger.info(f"Tags updated successfully for suggestion {request.suggestion_id}")

    return SuggestionResponse(
        id=str(updated_suggestion.id),
        title=updated_suggestion.title,
        tags=updated_suggestion.tag,
        suggestion=updated_suggestion.suggestion,
        updated_at=updated_suggestion.updated_at,
        created_at=updated_suggestion.created_at,
    )
