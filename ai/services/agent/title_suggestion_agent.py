from ai.utils.services import title_service


class TitleSuggestionAgent:
    async def run(self, input_text: str) -> list[str]:
        return await title_service.generate_title_suggestions(input_text)
