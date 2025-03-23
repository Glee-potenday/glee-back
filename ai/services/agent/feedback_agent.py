from ai.services.agent.summarizer_agent import SummarizerAgent
from ai.services.agent.reply_suggestion_agent import ReplySuggestionAgent


class FeedbackAgent:
    def __init__(self, min_length: int = 10, max_retries: int = 2) -> None:
        self.min_length = min_length
        self.max_retries = max_retries

    async def improve_reply(self, output: str, original_input: str, agent: ReplySuggestionAgent) -> str:
        retries = 0
        _output = output
        while len(_output.strip()) < self.min_length and retries < self.max_retries:
            improved_input = original_input + "\n추가 상세 설명 부탁해."
            reply = await agent.run(improved_input)
            _output = reply[0] if reply else output
            retries += 1
        return _output

    async def improve_summary(self, output: str, original_input: str, agent: SummarizerAgent) -> str:
        retries = 0
        _output = output
        while len(_output.strip()) < self.min_length and retries < self.max_retries:
            improved_input = original_input + "\n추가 상세 설명 부탁해."
            _output = await agent.run(improved_input)
            retries += 1

        return _output
