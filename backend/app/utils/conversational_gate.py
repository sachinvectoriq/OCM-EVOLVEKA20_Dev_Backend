"""
Conversational gate.

Lightweight pre-pipeline classifier that decides whether an incoming user
message is a normal conversational request (greeting, explicit identity)
or an actual query that should go to the RAG pipeline.

If conversational → return short reply and short-circuit pipeline.
If not conversational → return None and continue to RAG.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import Optional

from azure.identity import DefaultAzureCredential
from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient

from app.core.settings import Settings
from app.core.logger import Logger


_GATE_INSTRUCTIONS = """
You are a STRICT pre-classifier for an enterprise assistant.

Your ONLY job:
Decide whether the message is conversational OR NOT.

========================================================
CONVERSATIONAL = TRUE ONLY IF:
========================================================

1. GREETINGS / ACKNOWLEDGEMENTS ONLY
Examples:
hi, hello, hey, good morning, good evening, good afternoon,
bye, goodbye, thanks, thank you, ok, okay, got it, cool, nice

→ Respond with:
{
  "is_conversational": true,
  "reply": "short friendly response"
}

--------------------------------------------------------

2. EXPLICIT IDENTITY QUESTIONS ABOUT THE ASSISTANT
ONLY if the message directly refers to the assistant using:
- "you"
- "your"
- "yourself"
- OR explicitly says "assistant"

Examples:
- who are you
- what are you
- what can you do
- what can you help me with
- what is your purpose

→ Respond with:
I’m an enterprise onboarding assistant that helps you navigate internal onboarding systems, workflows, forms, and everything you need to get started smoothly..

========================================================
CRITICAL PRONOUN RULE (VERY IMPORTANT)
========================================================

You MUST NOT assume that:
- "it"
- "this"
- "that"
- "they"

refer to the assistant.

**If the message uses pronouns like "it", you MUST treat it as NOT conversational.

ONLY treat identity as assistant-related if the user explicitly uses:
"you / your / yourself / assistant"

Examples:
- "what does it do" → NOT conversational (RAG)
- "what is it" → NOT conversational (RAG)
- "what do you do" → conversational (identity)

========================================================
EVERYTHING ELSE → NOT CONVERSATIONAL
========================================================

All of the following MUST return:
{
  "is_conversational": false,
  "reply": ""
}

Includes:
- any question
- explanations
- definitions
- factual queries
- system questions (CRG, ESF, OTD, etc.)
- calculations
- instructions

Examples:
- what is CRG
- what does CRG do
- what does it do
- how does onboarding work
- explain anything

========================================================
OUTPUT FORMAT (STRICT JSON ONLY)
========================================================

{
  "is_conversational": true | false,
  "reply": "string or empty"
}
"""


class ConversationalGate:
    """Pre-pipeline classifier and direct responder."""

    def __init__(self, settings: Settings, logger: Logger) -> None:
        self.settings = settings
        self.logger = logger

        credential = DefaultAzureCredential() if settings.use_managed_identity else None

        chat_client = AzureOpenAIChatClient(
            credential=credential,
            endpoint=settings.azure_openai.endpoint,
            api_version=settings.azure_openai.api_version,
            deployment_name=settings.azure_openai.deployment_name,
        )

        self._agent = ChatAgent(
            chat_client=chat_client,
            name="ConversationalGate",
            instructions=_GATE_INSTRUCTIONS,
        )

        self.logger.info(
            f"ConversationalGate initialized with deployment: {settings.azure_openai.deployment_name}"
        )

    async def classify_and_reply(self, message: str) -> Optional[str]:
        text = (message or "").strip()
        if not text:
            return None

        try:
            result = await self._agent.run(
                messages=[ChatMessage(role=Role.USER, text=text)],
                max_tokens=200,
                temperature=0.0,
            )

            raw = (result.messages[-1].text or "").strip()

            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            parsed = json.loads(raw)

            if parsed.get("is_conversational") and parsed.get("reply"):
                return parsed["reply"]

            return None

        except Exception as e:
            self.logger.warning(
                f"ConversationalGate failed, falling back to RAG: {e}"
            )
            return None


_gate_instance: Optional[ConversationalGate] = None
_gate_lock = Lock()


def get_conversational_gate(settings: Settings, logger: Logger) -> ConversationalGate:
    global _gate_instance
    if _gate_instance is None:
        with _gate_lock:
            if _gate_instance is None:
                _gate_instance = ConversationalGate(settings, logger)
    return _gate_instance
