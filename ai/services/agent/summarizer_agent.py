from ai.utils.services import situation_service


class SummarizerAgent:
    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    async def run(self, input_text: str) -> str:
        retry = 0
        summary = ""
        while retry <= self.max_retries:
            summary = await situation_service.situation_summary(input_text)
            if len(summary.strip()) < 10 and retry < self.max_retries:
                input_text += "\n좀 더 자세히 요약해줘."
                retry += 1
                continue
            else:
                break
        return summary
