from dataclasses import asdict
from datetime import datetime
from typing import Any

import pymongo
from pymongo import ReturnDocument

from app.core.enums import SuggestionTagType
from app.suggester.suggester_document import SuggesterDocument, SuggesterDTO
from app.utils.mongo import db
from bson import ObjectId


class SuggesterCollection:
    """MongoDB `suggester` 컬렉션을 관리하는 클래스"""

    _collection = db["suggester"]

    @classmethod
    async def set_index(cls) -> None:
        """필요한 인덱스 설정"""
        await cls._collection.create_index([("user_id", pymongo.ASCENDING)])  # 사용자별 검색 최적화
        await cls._collection.create_index([("recommend", pymongo.ASCENDING)])  # ✅ recommend 필드 인덱싱 추가

    @classmethod
    async def create(cls, suggester_dto: SuggesterDTO) -> SuggesterDocument:
        """MongoDB에 데이터 저장 후 ObjectId 반환"""
        result = await cls._collection.insert_one(asdict(suggester_dto))
        tags = [SuggestionTagType(tag) for tag in suggester_dto.tag]
        return SuggesterDocument(
            user_id=suggester_dto.user_id,
            tag=tags,
            title=suggester_dto.title,
            suggestion=suggester_dto.suggestion,
            created_at=suggester_dto.created_at,
            updated_at=suggester_dto.updated_at,
            _id=result.inserted_id,
        )

    @classmethod
    async def get_by_id(cls, suggestion_id: str) -> dict[Any, Any] | None:
        """ID를 기반으로 데이터 조회"""
        return await cls._collection.find_one({"_id": ObjectId(suggestion_id)})

    @classmethod
    async def get_by_user(cls, user_id: ObjectId) -> list[dict[Any, Any]]:
        """사용자의 모든 추천 데이터 조회"""
        cursor = cls._collection.find({"user_id": user_id})
        return await cursor.to_list(length=100)

    @classmethod
    async def update(
        cls, suggestion_id: str, title: str, suggestion: str, tags: list[SuggestionTagType]
    ) -> SuggesterDocument:
        """추천 데이터 업데이트"""
        tags_str = [tag.value for tag in tags]

        result = await cls._collection.find_one_and_update(
            {"_id": ObjectId(suggestion_id)},
            {"$set": {"tag": tags_str, "title": title, "suggestion": suggestion, "updated_at": datetime.now()}},
            return_document=ReturnDocument.AFTER,  # 업데이트된 문서를 반환
        )

        return SuggesterDocument(
            title=result["title"],
            user_id=result["user_id"],
            tag=tags,
            suggestion=suggestion,
            created_at=result["created_at"],
            updated_at=result["updated_at"],
            _id=ObjectId(suggestion_id),
        )

    @classmethod
    async def delete(cls, suggestion_id: str) -> bool:
        """추천 데이터 삭제"""
        result = await cls._collection.delete_one({"_id": ObjectId(suggestion_id)})
        return result.deleted_count > 0

    @classmethod
    async def update_tag(cls, suggestion_id: str, tags: list[SuggestionTagType]) -> SuggesterDocument:
        tags_str = [tag.value for tag in tags]

        updated_doc = await cls._collection.find_one_and_update(
            {"_id": ObjectId(suggestion_id)},
            {"$set": {"tag": tags_str, "updated_at": datetime.now()}},
            return_document=ReturnDocument.AFTER,  # 업데이트된 문서를 반환
        )

        if not updated_doc:
            raise ValueError("Suggestion not found")

        return SuggesterDocument(
            title=updated_doc["title"],
            user_id=updated_doc["user_id"],
            tag=tags,
            suggestion=updated_doc["suggestion"],
            created_at=updated_doc["created_at"],
            updated_at=updated_doc["updated_at"],
            _id=updated_doc["_id"],
        )

    @classmethod
    async def get_recommend_documents(cls, query: str | None) -> list[dict[Any, Any]]:
        filter_criteria: dict[str, Any] = {"recommend": True}
        if query:
            filter_criteria["suggestion"] = {"$regex": query, "$options": "i"}
        cursor = cls._collection.find(filter_criteria)
        return await cursor.to_list(length=100)  # 최대 100개 가져오기

    @classmethod
    async def find_by_text(cls, query: str, user_id: ObjectId) -> list[dict[Any, Any]]:
        """본문에 특정 텍스트가 포함된 문서 검색 (suggestion, title 필드 모두)"""
        cursor = cls._collection.find(
            {
                "user_id": user_id,
                "$or": [
                    {"suggestion": {"$regex": query, "$options": "i"}},
                    {"title": {"$regex": query, "$options": "i"}},
                ],
            }
        )
        return await cursor.to_list(length=100)

    @classmethod
    async def count_by_user(cls, user_id: ObjectId) -> int:
        """특정 사용자의 제안 개수"""
        return await cls._collection.count_documents({"user_id": user_id})

    @classmethod
    async def count_recommend_documents(cls) -> int:
        """추천 제안 개수"""
        return await cls._collection.count_documents({"recommend": True})
