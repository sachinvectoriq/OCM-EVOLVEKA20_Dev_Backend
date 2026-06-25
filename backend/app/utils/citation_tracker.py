"""
Citation tracking utilities for source attribution.
Maintains source attribution throughout the pipeline.
"""
from typing import List, Dict, Optional

from app.models import Citation
from app.models.chat import RetrievedDocument
from app.core.logger import Logger


class CitationTracker:
    """
    Tracks citations and maintains source attribution throughout the pipeline.
    
    Ensures all retrieved documents are properly indexed and can be referenced
    in generated answers. Uses the API Citation schema as the single source of truth.
    """
    
    def __init__(self, logger: Logger):
        """Initialize citation tracker.
        
        Args:
            logger: Injected logging service.
        """
        self.logger = logger
        self.documents: Dict[str, RetrievedDocument] = {}
        self.citation_index = 0
    
    def add_documents(self, documents: List[RetrievedDocument]) -> None:
        """
        Add documents to citation tracker.
        
        Args:
            documents: List of retrieved documents
        """
        for doc in documents:
            if doc.content_id not in self.documents:
                self.documents[doc.content_id] = doc
                self.logger.debug(f"Added document to tracker: {doc.content_id}")
    
    def create_citations(
        self, 
        documents: List[RetrievedDocument]
    ) -> List[Citation]:
        """Create citations for all provided documents (vetted results).
        
        Args:
            documents: List of documents to cite (typically vetted results from reflection)
        
        Returns:
            List of Citation objects
        """
        citations = []
        
        for doc in documents:
            citation = Citation(
                document_id=doc.document_id,
                content_id=doc.content_id,
                content=doc.content,
                document_title=doc.title,
                page_number=doc.page_number                
            )
            citations.append(citation)
        
        return citations
    
    def get_document_by_id(self, document_id: str) -> Optional[RetrievedDocument]:
        """
        Get document by ID.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Retrieved document or None
        """
        return self.documents.get(document_id)
    
    def get_all_documents(self) -> List[RetrievedDocument]:
        """Get all tracked documents."""
        return list(self.documents.values())
    
    def get_document_count(self) -> int:
        """Get total number of tracked documents."""
        return len(self.documents)
