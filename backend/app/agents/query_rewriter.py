"""
Query Rewriter Agent for multi-strategy query expansion and rewriting.
Uses MAF ChatAgent for LLM interactions.
"""

from typing import List, Dict, Any, Optional
from azure.identity import DefaultAzureCredential

from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient
from opentelemetry import trace

from app.core.settings import Settings
from app.core.logger import Logger
from app.core.language import Language
from app.prompts import QueryRewriterPrompts
from app.prompts.localized import get_prompts
from app.models.chat import RewrittenQuery


class QueryRewriter:
    """
    Agent for HyDE (Hypothetical Document Embedding) query generation.
    
    Generates hypothetical document passages for semantic search, creating
    search queries that represent what the answer content might look like
    rather than traditional keyword-based queries.
    """
    
    def __init__(self, settings: Settings, logger: Logger):
        """
        Initialize the query rewriter agent using MAF ChatAgent.
        
        Args:
            settings: Application settings with Azure AI configuration
            logger: Injected logging service
        """
        self.settings = settings
        self.logger = logger
        self.tracer = trace.get_tracer("QueryRewriterAgent")
        
        # Initialize MAF ChatAgent
        credential = DefaultAzureCredential() if settings.use_managed_identity else None
        
        chat_client = AzureOpenAIChatClient(
            credential=credential,
            endpoint=settings.azure_openai.endpoint,
            api_version=settings.azure_openai.api_version,
            deployment_name=settings.azure_openai.deployment_name
        )

        # One ChatAgent per supported language so HyDE generation produces
        # passages in the same language as the corpus.
        self._chat_client = chat_client
        self._agents: Dict[Language, ChatAgent] = {
            Language.ENGLISH: ChatAgent(
                chat_client=chat_client,
                name="QueryRewriterAgent",
                instructions=get_prompts(Language.ENGLISH).hyde_system,
            ),
            Language.FRENCH_CANADIAN: ChatAgent(
                chat_client=chat_client,
                name="QueryRewriterAgent_FR",
                instructions=get_prompts(Language.FRENCH_CANADIAN).hyde_system,
            ),
        }
        self.agent = self._agents[Language.ENGLISH]
        
        self.logger.info(f"QueryRewriter initialized with MAF ChatAgent: {settings.azure_openai.deployment_name}")
    
    async def generate_hyde_search_query(
        self,
        user_query: str,
        search_history: Optional[List[Dict[str, Any]]] = None,
        previous_reviews: Optional[List[str]] = None,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate HyDE search query in the requested language."""
        lang = Language.from_value(language)
        self.logger.info(
            f"Generating HyDE search query ({lang.short_code}) for: {user_query[:100]}..."
        )
        
        # Build context for prompt
        context_parts = [f"User Question: {user_query}"]
        
        # Provide prior turns so follow-up questions (pronouns, ellipsis,
        # "which stage?", "how do I configure it?") can be resolved into a
        # self-contained topic before generating the hypothetical passage.
        # Self-contained questions must IGNORE this history (enforced in the
        # HyDE system prompt) so standalone queries are not contaminated.
        if conversation_history:
            history_lines = []
            for msg in conversation_history:
                role = (msg.get("role") or "").strip().lower()
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                speaker = "User" if role == "user" else "Assistant"
                history_lines.append(f"{speaker}: {content}")
            if history_lines:
                context_parts.append("\\n\\n### Conversation History (for follow-up resolution ONLY) ###")
                context_parts.extend(history_lines)
                context_parts.append(
                    "\\nIf the User Question above is self-contained, IGNORE this history entirely. "
                    "Only use it when the question depends on earlier turns to be understood."
                )
        
        # Add search history for subsequent attempts
        if search_history and previous_reviews:
            context_parts.append("\\n\\n### Previous Search Attempts ###")
            for i, (search, review) in enumerate(zip(search_history, previous_reviews), 1):
                context_parts.append(f"\\n<Attempt {i}>")
                context_parts.append(f"Query: {search.get('query', '')}")
                context_parts.append(f"Review: {review}")
                context_parts.append("</Attempt>")
            
            context_parts.append("\\n\\nCRITICAL: Since this is NOT the first search, you MUST diversify your approach:")
            context_parts.append("- Use different terminology, synonyms, or technical vs. layman terms")
            context_parts.append("- Focus on different aspects, time periods, or perspectives")
            context_parts.append("- Explore related concepts, causes, effects, or stakeholder viewpoints")
        
        context_parts.append("\\n\\nGenerate a hypothetical paragraph of what you expect to find in the target documents.")
        context_parts.append("Make it sound like the actual content, NOT like a search query.")
        
        try:
            context_text = '\n'.join(context_parts)
            user_prompt = f"\n{context_text}"
            
            rewritten_query = await self._call_llm(user_prompt, language=lang)
            
            self.logger.info(f"Generated HyDE query: {rewritten_query.hypothetical_passage[:150]}...")
            self.logger.info(f"Reasoning: {rewritten_query.reasoning}")
            
            return rewritten_query.hypothetical_passage
            
        except Exception as e:
            self.logger.error(f"HyDE generation failed: {e}")
            # Fallback to original query
            return user_query
    
    async def _call_llm(self, user_prompt: str, language: Language = Language.ENGLISH) -> RewrittenQuery:
        """Call LLM for query rewriting using the language-specific MAF ChatAgent."""
        message = ChatMessage(role=Role.USER, text=user_prompt)
        agent = self._agents.get(language, self._agents[Language.ENGLISH])
        result = await agent.run(
            messages=[message],
            response_format=RewrittenQuery,
            max_tokens=500,
            temperature=0.3
        )
        return RewrittenQuery.model_validate_json(result.messages[-1].text)
