from typing import Dict, Union, List

from ai.services.agent.feedback_agent import FeedbackAgent
from ai.services.agent.ocr_agent import OcrAgent
from ai.services.agent.reply_suggestion_agent import ReplySuggestionAgent
from ai.services.agent.style_analysis_agent import StyleAnalysisAgent
from ai.services.agent.summarizer_agent import SummarizerAgent
from ai.services.agent.title_suggestion_agent import TitleSuggestionAgent


class OrchestratorAgent:
    def __init__(self):
        self.ocr_agent = OcrAgent()
        self.summarizer_agent = SummarizerAgent()
        self.title_agent = TitleSuggestionAgent()
        self.reply_agent_old = ReplySuggestionAgent(variant="old")
        self.reply_agent_new = ReplySuggestionAgent(variant="new")
        self.style_agent = StyleAnalysisAgent()
        self.feedback_agent = FeedbackAgent()
        # self.video_service = VideoSearchService()

    async def run_reply_mode(self, input_text: str) -> Dict[str, Union[str, List[str]]]:
        # 상황 요약 생성
        summary = await self.summarizer_agent.run(input_text)
        summary = self.feedback_agent.check_and_improve(summary, input_text, self.summarizer_agent)

        # 제목 생성
        titles = await self.title_agent.run(summary)

        # 답장 제안 생성 (기본)
        replies = await self.reply_agent_old.run(summary)
        replies = [self.feedback_agent.check_and_improve(reply, summary, self.reply_agent_old) for reply in replies]

        return {
            "situation": summary,
            "accent": "기본 말투",
            "purpose": "일반 답변",
            "titles": titles,
            "replies": replies,
        }

    async def run_reply_with_video_info(self, input_text: str) -> Dict[str, Union[str, List[str]]]:
        """
        입력 텍스트를 기반으로 관련 자막 정보를 검색하고, 이를 답변 생성에 활용합니다.

        Args:
            input_text: 사용자 입력 텍스트

        Returns:
            생성된 답변 정보를 담은 딕셔너리
        """
        # 상황 요약 생성
        summary = await self.summarizer_agent.run(input_text)
        summary = self.feedback_agent.check_and_improve(summary, input_text, self.summarizer_agent)

        # 관련 자막 정보 검색
        video_info = await self.video_service.get_most_relevant_content(summary)

        # 자막 정보가 있는 경우, 이를 답변 생성에 활용
        if video_info["source"] != "error" and video_info["source"] != "no_results":
            # 자막 정보 추출
            video_title = video_info["video_title"]
            video_url = video_info["video_url"]
            transcript = video_info["transcript"]
            similarity = video_info.get("similarity", 0.0)

            # 자막 정보를 포함한 입력 생성
            enhanced_input = f"""
상황: {summary}

참고 자료:
제목: {video_title}
출처: {video_url}
내용: {transcript[:500]}...  # 너무 길지 않게 앞부분만 사용
"""

            # 제목 생성
            titles = await self.title_agent.run(summary)

            # 자막 정보를 포함한 답변 생성
            replies = await self.reply_agent_new.run(enhanced_input)
            replies = [
                self.feedback_agent.check_and_improve(reply, enhanced_input, self.reply_agent_new) for reply in replies
            ]

            return {
                "situation": summary,
                "accent": "기본 말투",
                "purpose": "일반 답변",
                "titles": titles,
                "replies": replies,
                "reference": {"video_title": video_title, "video_url": video_url, "similarity": similarity},
            }
        else:
            # 자막 정보가 없는 경우, 기본 답변 생성
            titles = await self.title_agent.run(summary)
            replies = await self.reply_agent_old.run(summary)
            replies = [self.feedback_agent.check_and_improve(reply, summary, self.reply_agent_old) for reply in replies]

            return {
                "situation": summary,
                "accent": "기본 말투",
                "purpose": "일반 답변",
                "titles": titles,
                "replies": replies,
            }

    async def run_style_mode(self, input_text: str) -> Dict[str, Union[str, List[str]]]:
        # 스타일 분석 (상황, 말투, 용도 추출)
        style_result, situation, tone, usage = await self.style_agent.run(input_text)
        style_result = self.feedback_agent.check_and_improve(style_result, input_text, self.style_agent)

        # 제목 제안 생성
        titles = await self.title_agent.run(situation)

        # 답변 제안 생성 (말투, 용도 정보 활용)
        detailed_input = f"상황: {situation}\n말투: {tone}\n용도: {usage}"
        replies = await self.reply_agent_new.run(detailed_input)
        replies = [
            self.feedback_agent.check_and_improve(reply, detailed_input, self.reply_agent_new) for reply in replies
        ]

        return {
            "situation": situation,
            "accent": tone,
            "purpose": usage,
            "titles": titles,
            "replies": replies,
            "style_analysis": style_result,
        }

    async def run_manual_mode(
        self, situation: str, accent: str, purpose: str, details: str
    ) -> Dict[str, Union[str, List[str]]]:
        # 입력 정보에 기반하여 전체 프롬프트 생성 (수동 입력으로 받을 경우)
        detailed_input = f"상황: {situation}\n말투: {accent}\n용도: {purpose}\n추가 설명: {details}"

        # 제목 제안 생성
        titles = await self.title_agent.run(situation)

        # 답변 제안 생성 (말투, 용도, 추가 설명 정보 활용)
        replies = await self.reply_agent_new.run(detailed_input)
        replies = [
            self.feedback_agent.check_and_improve(reply, detailed_input, self.reply_agent_new) for reply in replies
        ]

        return {
            "situation": situation,
            "accent": accent,
            "purpose": purpose,
            "details": details,
            "titles": titles,
            "replies": replies,
        }

    async def run_manual_mode_extended(
        self, suggestion: str, length: str, add_description: str
    ) -> Dict[str, Union[str, List[str]]]:

        suggestion_input = f"수정하고 싶은 답장: {suggestion}\n"

        if length:
            suggestion_input += f"원하는 답장 길이: {length}\n"
        if add_description:
            suggestion_input += f"추가 요청: {add_description}\n"

        suggestion_input += "위 내용을 바탕으로 자연스럽게 답장을 수정해서 작성해줘."

        # 제목 제안 생성(suggestion_input에서 suggestion으로 수정)
        titles = await self.title_agent.run(suggestion)

        # 답변 제안 생성
        replies = await self.reply_agent_new.run(suggestion_input)
        replies = [
            self.feedback_agent.check_and_improve(reply, suggestion_input, self.reply_agent_new) for reply in replies
        ]

        return {
            "titles": titles,
            "replies": replies,
        }
