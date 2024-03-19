from enum import Enum


class CitationChatMode(str, Enum):
    """Chat Engine Modes with Citation."""

    CONDENSE_QUESTION = "citation_condense_question"
    """Corresponds to `CitationCondenseQuestionChatEngine`.

    First generate a standalone question from conversation context and last message,
    then query the query engine for a response.
    """

    CONTEXT = "citation_context"
    """Corresponds to `CitationContextChatEngine`.

    First retrieve text from the index using the user's message, then use the context
    in the system prompt to generate a response.
    """

    CONDENSE_PLUS_CONTEXT = "citation_condense_plus_context"
    """Corresponds to `CitationCondensePlusContextChatEngine`.

    First condense a conversation and latest user message to a standalone question.
    Then build a context for the standalone question from a retriever,
    Then pass the context along with prompt and user message to LLM to generate a response.
    """
