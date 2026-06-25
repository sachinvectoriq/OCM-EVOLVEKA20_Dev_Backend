"""
Cosmos DB conversation history service for manual session management.

Key points:
- Manages conversation history using session_id + user_id pattern
- Uses Hierarchical Partition Key (HPK): [user_id, session_id]
- Provides methods needed by ChatService for agentic RAG workflows
- Uses async Cosmos client (azure.cosmos.aio) to avoid blocking

Schema (Cosmos item) matches ChatHistoryItem:
  id: str                    # unique = f"{user_id}_{session_id}_{message_id}"
  user_id: str               # partition key level 1
  session_id: str            # partition key level 2
  timestamp: str (ISO 8601)  # stored as string for reliable ordering
  serialized_message: str    # JSON with message details
  message_text: Optional[str]
  message_id: str
  role: str

Container requirements:
- Partition key: Hierarchical [/user_id, /session_id]
- You can optionally enable TTL
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import re

from azure.cosmos import PartitionKey, exceptions
from azure.cosmos.aio import CosmosClient

from agent_framework import ChatMessage, Role

from app.models.chat import ChatHistoryItem
from app.core.logger import Logger


class IConversationService(ABC):
    """Interface for conversation history service operations."""

    @abstractmethod
    async def get_conversation_history(
        self,
        session_id: str,
        user_id: str,
        max_messages: Optional[int] = None,
    ) -> List[ChatMessage]:
        """Retrieve conversation history with sanitized text (for LLM context).
        
        Returns messages with citations removed from text. Use this when passing
        conversation history to the LLM for generating responses to avoid cross-turn leakage.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            max_messages: Optional limit on number of messages to return
            
        Returns:
            List of ChatMessage with sanitized text in chronological order (oldest -> newest)
        """
        pass

    @abstractmethod
    async def get_conversation_history_with_citations(
        self,
        session_id: str,
        user_id: str,
        max_messages: Optional[int] = None,
    ) -> List[ChatMessage]:
        """Retrieve conversation history with full text including citations.
        
        Returns messages with original text including citation markers and metadata.        
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            max_messages: Optional limit on number of messages to return
            
        Returns:
            List of ChatMessage with full text and citations in chronological order (oldest -> newest)
        """
        pass

    @abstractmethod
    async def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to conversation history.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata to include in serialized_message
        """
        pass

    @abstractmethod
    async def list_user_sessions(
        self,
        user_id: str,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List all sessions for a user.
        
        Args:
            user_id: The user identifier
            max_results: Optional limit on number of sessions to return
            
        Returns:
            List of session dictionaries with session_id and metadata
        """
        pass

    @abstractmethod
    async def delete_session(
        self,
        session_id: str,
        user_id: str,
    ) -> None:
        """Delete all messages in a session.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
        """
        pass

    @abstractmethod
    async def clear_user_history(
        self,
        user_id: str,
    ) -> int:
        """Delete all conversation history for a user across all sessions.
        
        Args:
            user_id: The user identifier
            
        Returns:
            Number of messages deleted
        """
        pass


class ConversationService(IConversationService):
    """
    Cosmos DB conversation history service for manual session management.
    
    Provides methods needed by ChatService for agentic RAG workflows.
    Uses Hierarchical Partition Key (HPK): [user_id, session_id] for better scalability.
    """
    
    # Pre-compiled regex patterns for efficient citation stripping
    _NUMERIC_CITATION_PATTERN = re.compile(r'\[\d+\]')
    _CONTENT_ID_CITATION_PATTERN = re.compile(r'\{[^}]+\}')
    _MULTI_SPACE_PATTERN = re.compile(r' {2,}')
    _SPACE_NEWLINE_PATTERN = re.compile(r' ?\n ?')  # Handles both space before and after newline
    _SPACE_BEFORE_PUNCT_PATTERN = re.compile(r' ([.,])')  # Space before period or comma
    _MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')

    def __init__(
        self,
        cosmos_client: CosmosClient,
        database_name: str,
        container_name: str,
        logger: Optional[Logger] = None,
    ):
        self.logger = logger or Logger()
        self.client = cosmos_client
        self.db = cosmos_client.get_database_client(database_name)
        self.container = self.db.get_container_client(container_name)

    def _make_partition_key(self, user_id: str, session_id: str) -> List[str]:
        """Create HPK partition key value: [user_id, session_id]."""
        return [user_id, session_id]

    @staticmethod
    def is_not_found_error(ex: Exception) -> bool:
        """Check if exception is a Cosmos DB 404 Not Found error."""
        return isinstance(ex, exceptions.CosmosHttpResponseError) and getattr(ex, "status_code", None) == 404

    @classmethod
    def sanitize_message_text(cls, text: str) -> str:
        """Sanitize message text by removing citation patterns before saving to conversation history.
        
        Removes both inline numeric citations [1], [2], etc. and content ID citations
        {Content ID} to prevent citation leakage across conversation turns.
        
        Args:
            text: Text that may contain citation patterns
        
        Returns:
            Text with all citation patterns removed and whitespace cleaned up
        """
        # Remove citations using pre-compiled patterns
        result = cls._NUMERIC_CITATION_PATTERN.sub('', text)
        result = cls._CONTENT_ID_CITATION_PATTERN.sub('', result)
        
        # Clean up whitespace issues using pre-compiled patterns
        result = cls._MULTI_SPACE_PATTERN.sub(' ', result)           # Multiple spaces → single space
        result = cls._SPACE_NEWLINE_PATTERN.sub('\n', result)        # Space before/after newline → just newline
        result = cls._SPACE_BEFORE_PUNCT_PATTERN.sub(r'\1', result)  # Space before punctuation → just punctuation
        result = cls._MULTI_NEWLINE_PATTERN.sub('\n\n', result)      # Multiple newlines → max 2
        
        return result.strip()

    def _role_to_str(self, role: Any) -> str:
        """Convert Role enum or string to string."""
        if hasattr(role, "value"):
            return str(role.value)
        return str(role)

    def _payload_to_chat_message(
        self, 
        payload: Dict[str, Any], 
        text_override: Optional[str] = None
    ) -> ChatMessage:
        """Convert serialized payload back to ChatMessage.
        
        Args:
            payload: Serialized message data
            text_override: Optional text to use instead of payload text (for sanitized text)
        
        Returns:
            ChatMessage object
        """
        role_str = (payload.get("role") or "user").lower()

        role_map = {
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
            "system": Role.SYSTEM,
        }
        if hasattr(Role, "TOOL"):
            role_map["tool"] = Role.TOOL

        # Use text_override if provided (for sanitized text), otherwise use payload text
        text = text_override if text_override is not None else (payload.get("text", "") or "")

        return ChatMessage(
            role=role_map.get(role_str, Role.USER),
            text=text,
            id=payload.get("id")            
        )

    async def get_conversation_history(
        self,
        session_id: str,
        user_id: str,
        max_messages: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Retrieve conversation history with sanitized text (for LLM context).
        
        Returns messages with citations removed from text. Use this when passing
        conversation history to the LLM for generating responses.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            max_messages: Optional limit on number of messages to return
            
        Returns:
            List of ChatMessage with sanitized text in chronological order (oldest -> newest)
        """
        partition_key = self._make_partition_key(user_id, session_id)
        
        query = (
            "SELECT * FROM c "
            "ORDER BY c.timestamp ASC"
        )

        items: List[Dict[str, Any]] = []
        try:
            async for it in self.container.query_items(
                query=query,
                partition_key=partition_key,
            ):
                items.append(it)
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"[ConversationService] query_items failed: {e}")
            return []

        # Apply max_messages limit if specified
        if max_messages is not None and max_messages > 0 and len(items) > max_messages:
            items = items[-max_messages:]

        # Convert to ChatMessage objects using sanitized message_text field
        messages: List[ChatMessage] = []
        for it in items:
            try:
                payload = json.loads(it["serialized_message"])
            except Exception:
                # Fallback if serialized_message is corrupted
                payload = {
                    "id": it.get("message_id"),
                    "role": it.get("role", "user"),
                }
            
            # Use message_text field (guaranteed sanitized) for LLM context
            sanitized_text = it.get("message_text", "") or ""
            messages.append(self._payload_to_chat_message(payload, text_override=sanitized_text))

        return messages

    async def get_conversation_history_with_citations(
        self,
        session_id: str,
        user_id: str,
        max_messages: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Retrieve conversation history with full text including citations (for UI display).
        
        Returns messages with original text including citation markers and metadata.
        Use this when displaying conversation history in the UI.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            max_messages: Optional limit on number of messages to return
            
        Returns:
            List of ChatMessage with full text and citations in chronological order (oldest -> newest)
        """
        partition_key = self._make_partition_key(user_id, session_id)
        
        query = (
            "SELECT * FROM c "
            "ORDER BY c.timestamp ASC"
        )

        items: List[Dict[str, Any]] = []
        try:
            async for it in self.container.query_items(
                query=query,
                partition_key=partition_key,
            ):
                items.append(it)
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"[ConversationService] query_items failed: {e}")
            return []

        # Apply max_messages limit if specified
        if max_messages is not None and max_messages > 0 and len(items) > max_messages:
            items = items[-max_messages:]

        # Convert to ChatMessage objects using serialized_message (contains citations)
        messages: List[ChatMessage] = []
        for it in items:
            try:
                payload = json.loads(it["serialized_message"])
            except Exception:
                # Fallback if serialized_message is corrupted
                payload = {
                    "id": it.get("message_id"),
                    "role": it.get("role", "user"),
                    "text": it.get("message_text", "") or "",
                }
            messages.append(self._payload_to_chat_message(payload))

        return messages

    async def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            role: Message role (user, assistant, system)
            content: Message content (citations will be automatically stripped)
            metadata: Optional metadata to include in serialized_message
        """
        message_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        # Sanitize message text by removing all citations before saving to prevent cross-turn citation leakage
        message_text = self.sanitize_message_text(content)

        # Build the serialized message payload
        payload = {
            "id": message_id,
            "role": role,    
            "text": message_text,
        }
        if metadata:
            payload.update(metadata)
        
        item = ChatHistoryItem(
            id=f"{user_id}_{session_id}_{message_id}",
            user_id=user_id,
            session_id=session_id,
            timestamp=now,
            serialized_message=json.dumps(payload, ensure_ascii=False),
            message_text=message_text,
            message_id=message_id,
            role=role,
        )

        # Convert to dict and format timestamp
        doc = item.model_dump()
        doc["timestamp"] = item.timestamp.isoformat()        

        try:
            await self.container.upsert_item(body=doc)
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"[ConversationService] upsert_item failed: {e}")
            raise

    async def list_user_sessions(
        self,
        user_id: str,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all sessions for a user.
        
        Args:
            user_id: The user identifier
            max_results: Optional limit on number of sessions to return
            
        Returns:
            List of session dictionaries with session_id and metadata
        """
        # Query distinct session_ids for this user
        # This is a cross-partition query because we need all sessions (different second-level partition keys)
        query = "SELECT DISTINCT c.session_id FROM c WHERE c.user_id = @user_id"
        parameters: List[Dict[str, Any]] = [{"name": "@user_id", "value": user_id}]

        session_ids: List[str] = []
        try:
            iterator = self.container.query_items(
                query=query,
                parameters=parameters,
            )

            async for it in iterator:
                if isinstance(it, str):
                    session_id = it
                else:
                    session_id = it.get("session_id")

                if session_id:
                    session_ids.append(session_id)
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"[ConversationService] list_user_sessions failed: {e}")
            return []

        # Apply max_results if specified
        if max_results is not None and max_results > 0:
            session_ids = session_ids[:max_results]

        # Build session list
        sessions = []
        for session_id in session_ids:
            sessions.append({
                "session_id": session_id,
            })

        return sessions

    async def delete_session(
        self,
        session_id: str,
        user_id: str,
    ) -> None:
        """
        Delete all messages in a session.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
        """
        partition_key = self._make_partition_key(user_id, session_id)
        
        query = "SELECT c.id FROM c"

        try:
            # Get all message IDs
            message_ids = []
            async for it in self.container.query_items(
                query=query,
                partition_key=partition_key,
            ):
                message_ids.append(it["id"])

            # Delete each message
            for msg_id in message_ids:
                await self.container.delete_item(
                    item=msg_id,
                    partition_key=partition_key,
                )
                
            self.logger.info(f"[ConversationService] Deleted {len(message_ids)} messages from session {session_id} for user {user_id}")
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"[ConversationService] delete_session failed: {e}")
            raise

    async def clear_user_history(
        self,
        user_id: str,
    ) -> int:
        """
        Delete all conversation history for a user across all sessions.
        
        Args:
            user_id: The user identifier
            
        Returns:
            Number of messages deleted
        """
        # With HPK, we can query by user_id prefix efficiently
        query = "SELECT c.id, c.user_id, c.session_id FROM c WHERE c.user_id = @user_id"
        parameters: List[Dict[str, Any]] = [{"name": "@user_id", "value": user_id}]

        deleted_count = 0
        try:
            # Collect all items to delete
            items_to_delete: List[tuple[str, List[str]]] = []
            iterator = self.container.query_items(
                query=query,
                parameters=parameters,
            )

            async for it in iterator:
                item_id = it.get("id")
                u_id = it.get("user_id")
                s_id = it.get("session_id")
                if item_id and u_id and s_id:
                    items_to_delete.append((item_id, [u_id, s_id]))

            # Delete each item using HPK partition key
            for item_id, pk in items_to_delete:
                await self.container.delete_item(
                    item=item_id,
                    partition_key=pk,
                )
                deleted_count += 1
                
            self.logger.info(f"[ConversationService] Deleted {deleted_count} messages for user {user_id}")
        except exceptions.CosmosHttpResponseError as e:
            self.logger.error(f"[ConversationService] clear_user_history failed: {e}")
            raise

        return deleted_count

