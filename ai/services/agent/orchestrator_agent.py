from typing import Dict, Union, List

from ai.services.agent.feedback_agent import FeedbackAgent
from ai.services.agent.ocr_agent import OcrAgent
from ai.services.agent.reply_suggestion_agent import ReplySuggestionAgent
from ai.services.agent.style_analysis_agent import StyleAnalysisAgent
from ai.services.agent.summarizer_agent import SummarizerAgent
from ai.services.agent.title_suggestion_agent import TitleSuggestionAgent


class OrchestratorAgent:
    def __init__(self) -> None:
        self.ocr_agent = OcrAgent()
        self.summarizer_agent = SummarizerAgent()
        self.title_agent = TitleSuggestionAgent()
        self.reply_agent_old = ReplySuggestionAgent(variant="old")
        self.reply_agent_new = ReplySuggestionAgent(variant="new")
        self.style_agent = StyleAnalysisAgent()
        self.feedback_agent = FeedbackAgent()

    async def run_reply_mode(self, input_text: str) -> tuple[list[str], list[str]]:
        # 상황 요약 생성
        summary = await self.summarizer_agent.run(input_text)
        feedback_summary = await self.feedback_agent.improve_summary(summary, input_text, self.summarizer_agent)

        if isinstance(feedback_summary, list):
            raise ValueError("feedback_summary Type Error")
        # 제목 생성
        titles = await self.title_agent.run(feedback_summary)

        # 답장 제안 생성 (기본)
        replies = await self.reply_agent_old.run(summary)
        feedback_replies = [
            await self.feedback_agent.improve_reply(reply, summary, self.reply_agent_old) for reply in replies
        ]

        if isinstance(feedback_replies, str):
            raise ValueError("feedback_replies Type Error")

        return titles, replies

    async def run_manual_mode(
        self, situation: str, accent: str, purpose: str, details: str
    ) -> tuple[list[str], list[str]]:
        # 입력 정보에 기반하여 전체 프롬프트 생성 (수동 입력으로 받을 경우)
        detailed_input = f"상황: {situation}\n말투: {accent}\n용도: {purpose}\n추가 설명: {details}"

        # 제목 제안 생성
        titles = await self.title_agent.run(situation)

        # 답변 제안 생성 (말투, 용도, 추가 설명 정보 활용)
        replies = await self.reply_agent_new.run(detailed_input)
        feedreplies = [
            await self.feedback_agent.improve_reply(reply, detailed_input, self.reply_agent_new) for reply in replies
        ]

        return titles, replies

    async def run_manual_mode_extended(
        self, suggestion: str, length: str, add_description: str
    ) -> tuple[list[str], list[str]]:

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
            await self.feedback_agent.improve_reply(reply, suggestion_input, self.reply_agent_new) for reply in replies
        ]

        return titles, replies
