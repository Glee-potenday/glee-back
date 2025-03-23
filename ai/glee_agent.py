from dotenv import load_dotenv
from ai.services.agent.ocr_agent import OcrAgent
from ai.services.agent.orchestrator_agent import OrchestratorAgent
from ai.services.agent.style_analysis_agent import StyleAnalysisAgent
from ai.services.agent.summarizer_agent import SummarizerAgent

# 프로젝트 루트 디렉토리를 Python 경로에 추가
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from app.suggester.suggester_dto import AiSuggestionDto

load_dotenv()  # .env 파일 로드


class GleeAgent:
    ocr_agent: OcrAgent = OcrAgent()
    summarizer_agent: SummarizerAgent = SummarizerAgent()
    style_agent: StyleAnalysisAgent = StyleAnalysisAgent()

    orchestrator_agent: OrchestratorAgent = OrchestratorAgent()

    @classmethod  # 실제적으로 사용되지 않는 메서드같습니다
    async def parse_suggestion(cls, suggestion: str) -> tuple[str, str]:
        """제안 텍스트에서 제목과 내용을 추출합니다."""
        title = ""
        content = suggestion

        # 콜론(:)이 있는지 확인하고 이후의 내용만 추출
        if ":" in suggestion:
            # 첫 번째 콜론을 기준으로 분할
            parts = suggestion.split(":", 1)
            if len(parts) > 1:
                # 콜론 이전 부분이 "제목"을 포함하는지 확인
                if "제목" in parts[0].lower():
                    content = ""  # 제목만 있는 경우 내용은 빈 문자열로 설정
                    title = parts[1].strip()
                else:
                    # 제목이 아닌 다른 콜론이 있는 경우 원래 내용 유지
                    content = suggestion

        return title, content

    # -------------------------------------------------------------------
    # [1] 이미지파일 (최대 4개) 입력 -> 상황을 뱉어내는 함수
    @classmethod
    async def analyze_situation(cls, image_files: list[tuple[str, bytes]]) -> str:
        if not image_files:
            raise ValueError("No image files provided.")

        # ocr 에이전트를 사용하여 텍스트 추출
        image_text = await cls.ocr_agent.run(image_files)

        # 상황 요약 에이전트를 사용하여 상황 분석
        situation_string = await cls.summarizer_agent.run(image_text)
        return situation_string

    # [2] 이미지파일 (최대 4개) 입력 -> 상황, 말투, 용도를 뱉어내는 함수
    @classmethod
    async def analyze_situation_accent_purpose(cls, image_files: list[tuple[str, bytes]]) -> tuple[str, str, str]:
        if not image_files:
            return "", "", ""

        # ocr 에이전트를 사용하여 텍스트 추출
        image_text = await cls.ocr_agent.run(image_files)

        # 스타일 분석 에이전트를 사용하여 스타일 분석
        _, situation, accent, purpose = await cls.style_agent.run(image_text)
        return situation, accent, purpose

    # -------------------------------------------------------------------
    # [3] 상황만을 기반으로 글 제안을 생성하는 함수
    @classmethod
    async def generate_suggestions_situation(cls, situation: str) -> AiSuggestionDto:
        title, suggestion = await cls.orchestrator_agent.run_reply_mode(situation)
        return AiSuggestionDto(titles=title, suggestions=suggestion)

    # -------------------------------------------------------------------
    # [4] 상황, 말투, 용도를 기반으로 글 제안을 생성하는 함수
    @classmethod
    async def generate_reply_suggestions_accent_purpose(
        cls, situation: str, accent: str, purpose: str
    ) -> AiSuggestionDto:

        title, suggestion = await cls.orchestrator_agent.run_manual_mode(situation, accent, purpose, "")
        return AiSuggestionDto(titles=title, suggestions=suggestion)

    # -------------------------------------------------------------------
    # [5] 상황, 말투, 용도, 상세 설명을 기반으로 글 제안을 생성하는 함수
    @classmethod
    async def generate_reply_suggestions_detail(
        cls, situation: str, accent: str, purpose: str, detailed_description: str
    ) -> AiSuggestionDto:

        title, suggestion = await cls.orchestrator_agent.run_manual_mode(
            situation, accent, purpose, detailed_description
        )
        return AiSuggestionDto(titles=title, suggestions=suggestion)

    # -------------------------------------------------------------------
    # [6] 상황, 말투, 용도, 상세 설명, 글 길이를 기반으로 글 제안을 생성하는 함수
    #  length : 짧게, 길게, 적당함 (short, long, moderate) 예정
    @classmethod
    async def generate_reply_suggestions_detail_length(
        cls, suggestion: str, length: str, add_description: str
    ) -> AiSuggestionDto:

        title, extend_suggestion = await cls.orchestrator_agent.run_manual_mode_extended(
            suggestion, length, add_description
        )
        return AiSuggestionDto(titles=title, suggestions=extend_suggestion)
