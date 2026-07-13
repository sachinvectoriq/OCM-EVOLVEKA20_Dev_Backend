"""
Conversational gate.

Lightweight pre-pipeline classifier that decides whether an incoming user
message is a normal conversational / meta request (greeting, identity
question, acknowledgement, "make it shorter", etc.) or an actual question
that should be sent to the RAG pipeline.

If the message is conversational, this module produces a short natural
reply directly via Azure OpenAI and the API route short-circuits the
pipeline — returning an empty citations / follow_ups payload.

If the message is a real query (any kind — even off-topic ones like
"distance between moon and sun"), this module returns ``None`` so the
caller proceeds with the normal agentic RAG flow.
"""

from __future__ import annotations

import json
import re
from threading import Lock
from typing import Optional

from azure.identity import DefaultAzureCredential
from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient

from app.core.settings import Settings
from app.core.logger import Logger


# ---------------------------------------------------------------------------
# Deterministic fast-path
#
# The LLM gate below is inherently flaky (a chatty model can wrap its JSON in
# prose, or occasionally mis-classify an obvious greeting), and on any failure
# ``classify_and_reply`` fails open — which pushes greetings into the RAG
# pipeline and returns a "no relevant documents" fallback. To keep the most
# common conversational inputs ("hi", "what can you do", "make it shorter")
# working reliably, we short-circuit them here without calling the LLM.
# ---------------------------------------------------------------------------

_GREETING_REPLY = (
    "Hi! I can help you find answers from the organization's onboarding "
    "documentation (topics include ESF, CRG, OTD, Bullhorn, forms and "
    "packages, and related processes). What would you like to know?"
)

_IDENTITY_REPLY = (
    "I'm an onboarding assistant. I answer questions from the organization's "
    "onboarding documentation — topics such as ESF, CRG, OTD, Bullhorn, forms "
    "and packages, and related processes. Ask me anything about those."
)

_META_REPLY = (
    "Sure — could you clarify which previous answer you'd like me to restyle? "
    "I don't have it in front of me here, so please paste it or ask your "
    "question again."
)

# Exact-match greetings / acknowledgements (after normalization).
_GREETING_EXACT = {
    "hi", "hii", "hiii", "hey", "heya", "hello", "helo", "hello there",
    "hi there", "hey there", "yo", "sup", "whatsup", "what's up", "whats up",
    "good morning", "good afternoon", "good evening", "greetings",
    "thanks", "thank you", "thank u", "thx", "ty", "much appreciated",
    "ok", "okay", "k", "kk", "got it", "cool", "nice", "great", "awesome",
    "bye", "goodbye", "see you", "see ya", "cya", "how are you",
    "how are you doing", "how's it going", "hows it going",
}

# Identity / capability questions about the assistant.
_IDENTITY_PATTERNS = [
    re.compile(r"^who\s+are\s+you\b"),
    re.compile(r"^what\s+are\s+you\b"),
    re.compile(r"^what\s+can\s+you\s+do\b"),
    re.compile(r"^what\s+do\s+you\s+do\b"),
    re.compile(r"^what\s+can\s+you\s+help\b"),
    re.compile(r"^how\s+can\s+you\s+help\b"),
    re.compile(r"^what('?s|\s+is)\s+your\s+(purpose|name)\b"),
    re.compile(r"^what\s+are\s+you\s+capable\s+of\b"),
    re.compile(r"^tell\s+me\s+about\s+yourself\b"),
]

# Meta directives on a previous answer.
_META_PATTERNS = [
    re.compile(r"^(make\s+it|be)\s+(shorter|concise|brief)\b"),
    re.compile(r"^(summarize|summarise)\s+(that|it|this)\b"),
    re.compile(r"^(expand|elaborate)\b"),
    re.compile(r"^(rephrase|reword)\b"),
    re.compile(r"^explain\s+(that|it|this)?\s*(simpler|again)\b"),
    re.compile(r"^give\s+me\s+bullet\s+points\b"),
]


def _fast_path_reply(text: str) -> Optional[str]:
    """Return a canned reply for obvious conversational inputs, else ``None``.

    This is a deterministic pre-classifier that never calls the LLM, so the
    most common greetings / identity / meta requests always get a friendly
    reply even when the LLM gate misbehaves.
    """
    # Normalize: lowercase, strip surrounding punctuation/whitespace, collapse spaces.
    normalized = text.strip().lower()
    normalized = re.sub(r"[!.?,]+$", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    if not normalized:
        return None

    if normalized in _GREETING_EXACT:
        return _GREETING_REPLY

    for pattern in _IDENTITY_PATTERNS:
        if pattern.search(normalized):
            return _IDENTITY_REPLY

    for pattern in _META_PATTERNS:
        if pattern.search(normalized):
            return _META_REPLY

    return None


def _extract_json_object(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from a possibly noisy LLM reply.

    Handles code fences and leading/trailing prose by locating the outermost
    ``{ ... }`` span before parsing.
    """
    if not raw:
        return None

    cleaned = raw.strip()
    # Strip code fences like ```json ... ```.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    # Try a direct parse first.
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to the outermost object span (rescues wrapped-in-prose replies).
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


_GATE_INSTRUCTIONS = """You are a strict pre-classifier for an enterprise onboarding assistant.

You have ONLY two allowed jobs. You must never go beyond them.

================================================================
ALLOWED CATEGORIES (respond directly with a short natural reply)
================================================================

1. GREETINGS / SMALL TALK / ACKNOWLEDGEMENTS
   Examples: "hi", "hello", "hey", "good morning", "how are you", "thanks",
   "thank you", "bye", "ok", "got it", "cool", "nice".
   → Reply: short, warm greeting/acknowledgement and invite an onboarding question.

2. IDENTITY / CAPABILITY QUESTIONS ABOUT YOU (the assistant)
   Examples: "who are you", "what are you", "what can you do",
   "what can you help me with", "how can you help", "what's your purpose".
   → Reply: briefly say you help answer questions from the organization's
     onboarding documentation (topics include ESF, CRG, OTD, Bullhorn,
     forms and packages, and related processes). 1–2 sentences.

3. META DIRECTIVES ON A PREVIOUS ANSWER (only when clearly meta)
   Examples: "make it shorter", "summarize that", "be concise", "expand",
   "explain simpler", "rephrase", "optimize this", "give me bullet points".
   → Reply: briefly ask the user to clarify which previous answer they want
     restyled (since you do not have it in front of you here).

================================================================
EVERYTHING ELSE → NOT YOUR JOB
================================================================

If the user is asking for ANY kind of factual, informational, procedural,
opinion, calculation, definition, geographic, mathematical, scientific,
historical, organizational, product, or general-knowledge content — even
trivially — you MUST refuse to answer it here and let it pass through.

You must NEVER answer:
- "how many districts in NYC"
- "distance between the moon and the sun"
- "capital of France"
- "what is python"
- "what is a CRG"
- "who maintains the ESF"
- "explain machine learning"
- "what's 2 + 2"
- ANY question that seeks knowledge, even if you know the answer.
- ANY question about the company's onboarding content (those belong to the pipeline).

For these, set is_conversational = false and leave reply empty.
Do NOT explain. Do NOT redirect. Do NOT apologize. Just classify as not conversational.

================================================================
HARD RULES
================================================================

- You are NOT a general chatbot. You do not have knowledge. You do not answer questions.
- Only categories 1, 2, 3 above produce a reply. EVERYTHING else returns an empty reply.
- Never include citations. Never use markdown. Plain text only.
- Keep replies to 1–2 short sentences.
- If unsure whether something is conversational, classify as NOT conversational.

================================================================
OUTPUT FORMAT (STRICT)
================================================================

Return ONLY valid JSON in this exact shape, no extra text:

{
  "is_conversational": true | false,
  "reply": "the natural reply when is_conversational is true, otherwise empty string"
}
"""


class ConversationalGate:
    """Pre-pipeline classifier and direct responder for conversational input."""

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
        """
        Classify the user's message.

        Returns:
            A natural-language reply string if the message is conversational/meta
            (caller should short-circuit the pipeline and emit this as the answer).
            ``None`` if the message is a real query and should continue to the
            RAG pipeline.
        """
        text = (message or "").strip()
        if not text:
            return None

        # Deterministic fast-path for obvious greetings / identity / meta.
        # This never depends on the LLM, so these always get a friendly reply.
        fast_reply = _fast_path_reply(text)
        if fast_reply:
            self.logger.info("ConversationalGate fast-path matched (no LLM call).")
            return fast_reply

        try:
            result = await self._agent.run(
                messages=[ChatMessage(role=Role.USER, text=text)],
                max_tokens=200,
                temperature=0.2,
            )
            raw = (result.messages[-1].text or "").strip()
            parsed = _extract_json_object(raw)
            if parsed is None:
                self.logger.warning(
                    "ConversationalGate could not parse JSON from LLM reply; "
                    f"falling through to pipeline. Raw reply: {raw[:200]!r}"
                )
                return None
            is_conv = bool(parsed.get("is_conversational", False))
            reply = (parsed.get("reply") or "").strip()
            if is_conv and reply:
                return reply
            return None
        except Exception as e:
            # On any failure, fail open: treat as a real query so the pipeline still runs.
            self.logger.warning(
                f"ConversationalGate classification failed ({type(e).__name__}: {e}), "
                "falling through to pipeline."
            )
            return None


_gate_instance: Optional[ConversationalGate] = None
_gate_lock = Lock()


def get_conversational_gate(settings: Settings, logger: Logger) -> ConversationalGate:
    """Return a process-wide singleton ConversationalGate."""
    global _gate_instance
    if _gate_instance is None:
        with _gate_lock:
            if _gate_instance is None:
                _gate_instance = ConversationalGate(settings, logger)
    return _gate_instance
