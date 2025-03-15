# Glee
답?답! 답답한 상황에서 답을 내어줄게 - 백엔드 레포지토리

https://www.glee-message-maker.store/


# 기술 스택
- Python, FastAPI, MongoDB
- Nginx, Docker, Github Action
- CLOVA OCR, CLOVA STUDIO
- Kakao OAuth
# 핵심 기능
### 1. 사용자 관리
   - 카카오 소셜 로그인과 JWT 기반 토큰 관리
   - 프로필 조회

### 2. AI 글 제안 기능
   - AI 기반 글 생성
     - OCR 기반 대화 사진 속 상황, 말투, 목적 분석
     - 분석 상황에서의 AI 글 제안 생성
     - 사용자 요청에 기반하여 글 수정
   - 사용자 글 제안 관리
     - 최근 생성 글 제안 & 저장한 글 조회 수정 삭제
     - 태그 및 추천 기능
# 아키텍쳐
![시스템 아키텍쳐](https://github.com/user-attachments/assets/5c41c5ba-5b5e-4b4a-bd3c-a80694c1077e)


# Install
```
poetry install
```
# Run
```
poetry run uvicorn app.main:app
```


# Test
```
poetry run pytest
```
