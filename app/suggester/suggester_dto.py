from dataclasses import dataclass


@dataclass(frozen=True)
class AiSuggestionDto:
    titles: list[str]
    suggestions: list[str]
