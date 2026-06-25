"""
Logging service with Application Insights integration.

This module provides a centralized logging service that can be injected
into other services via dependency injection.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional
import time

from azure.monitor.opentelemetry import configure_azure_monitor
from agent_framework.observability import enable_instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode


class Logger:
    """
    Centralized logging service with Application Insights integration.
    
    Provides structured logging capabilities with automatic telemetry
    to Application Insights when configured.
    
    Features:
    - Console logging with formatted output
    - Application Insights integration via OpenTelemetry
    - Structured logging with extra context
    - Automatic exception tracking
    - Performance measurement helpers
    
    Example:
    --------
    ```python
    # Via dependency injection
    def __init__(self, logger: Logger):
        self.logger = logger
    
    # Usage
    self.logger.info("Processing started", extra={"blob_name": "test.pdf"})
    self.logger.error("Processing failed", exc_info=True)
    ```
    """
    
    _app_insights_configured: bool = False
    
    def __init__(self, name: str = "knowledge_assistant") -> None:
        """
        Initialize the logging service.
        
        Parameters
        ----------
        name : str
            Logger name (default: "knowledge_assistant")
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self._configure_app_insights_once()
        self.tracer = trace.get_tracer(name)
    
    @classmethod
    def _configure_app_insights_once(cls) -> None:
        """
        Configure Application Insights OpenTelemetry (singleton pattern).
        
        This ensures configure_azure_monitor() is only called once
        across all Logger instances.
        """
        if cls._app_insights_configured:
            return
        
        # Configure console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # Override any existing configuration
        )
                                    
        # Configure Application Insights if connection string is available
        conn_str = os.getenv("APPINSIGHTS_CONNECTION_STRING")
        if conn_str:
            try:
                # Create Resource with service metadata
                resource = Resource.create({
                    "service.name": "knowledge-assistant",
                    "service.version": "1.0.0",
                    "deployment.environment": os.getenv("ENVIRONMENT", "development"),
                })
                
                # Set up TracerProvider with Resource
                tracer_provider = TracerProvider(resource=resource)
                trace.set_tracer_provider(tracer_provider)
                
                # Configure Azure Monitor with OpenTelemetry                
                configure_azure_monitor(
                    connection_string=conn_str,
                    enable_live_metrics=True,
                    resource=resource
                )
                
                # Enable MAF instrumentation for ChatAgent and tools                
                enable_instrumentation(
                    enable_sensitive_data=os.getenv("ENABLE_SENSITIVE_DATA", "false").lower() == "true"
                )
                
                logging.getLogger(__name__).info(
                    "✓ Application Insights OpenTelemetry configured successfully"
                )
                    
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"✗ Failed to configure Application Insights: {str(e)}"
                )
                logging.getLogger(__name__).warning(
                    "Continuing without Application Insights telemetry"
                )
        else:
            logging.getLogger(__name__).warning(
                "⚠ APPINSIGHTS_CONNECTION_STRING not set - "
                "Application Insights telemetry disabled (console logging only)"
            )
        
        cls._app_insights_configured = True
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Log an info message.
        
        Parameters
        ----------
        message : str
            Log message
        extra : dict, optional
            Additional structured data for Application Insights
        **kwargs
            Additional logging parameters (exc_info, stack_info, etc.)
        """
        self.logger.info(message, extra=extra or {}, **kwargs)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Log a warning message.
        
        Parameters
        ----------
        message : str
            Log message
        extra : dict, optional
            Additional structured data for Application Insights
        **kwargs
            Additional logging parameters
        """
        self.logger.warning(message, extra=extra or {}, **kwargs)
    
    def error(
        self, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Log an error message.
        
        Parameters
        ----------
        message : str
            Log message
        extra : dict, optional
            Additional structured data for Application Insights
        exc_info : bool
            Include exception information (default: True)
        **kwargs
            Additional logging parameters
        """
        self.logger.error(message, extra=extra or {}, exc_info=exc_info, **kwargs)
    
    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Log an exception with full stack trace.
        
        Parameters
        ----------
        message : str
            Log message
        extra : dict, optional
            Additional structured data for Application Insights
        **kwargs
            Additional logging parameters
        """
        self.logger.exception(message, extra=extra or {}, **kwargs)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Log a debug message.
        
        Parameters
        ----------
        message : str
            Log message
        extra : dict, optional
            Additional structured data for Application Insights
        **kwargs
            Additional logging parameters
        """
        self.logger.debug(message, extra=extra or {}, **kwargs)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Log a critical message.
        
        Parameters
        ----------
        message : str
            Log message
        extra : dict, optional
            Additional structured data for Application Insights
        **kwargs
            Additional logging parameters
        """
        self.logger.critical(message, extra=extra or {}, **kwargs)
    
    def add_span_attributes(self, **attributes: Any) -> None:
        """
        Add custom attributes to the current OpenTelemetry span.
        
        This enriches MAF-generated spans with business context like
        user IDs, query characteristics, document counts, etc.
        
        Parameters
        ----------
        **attributes
            Key-value pairs to add as span attributes
        
        Example
        -------
        ```python
        logger.add_span_attributes(
            user_id="user123",
            query_length=45,
            document_count=5
        )
        ```
        """
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes: Any):
        """
        Context manager for tracing non-MAF operations (Azure Search, Cosmos DB, etc.).
        
        MAF automatically instruments agent.run() calls, but use this for
        external service calls that need explicit tracing.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation (e.g., "azure_search_query", "cosmos_upsert")
        **attributes
            Additional span attributes
        
        Example
        -------
        ```python
        with logger.trace_operation("azure_search", index="docs", query_type="vector"):
            results = await search_client.search(query)
        ```
        """
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add custom attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_ms", duration * 1000)
    
    def log_operation_start(self, operation: str, **context: Any) -> None:
        """
        Log the start of an operation with context.
        
        Parameters
        ----------
        operation : str
            Name of the operation
        **context
            Additional context key-value pairs
        """
        extra: Dict[str, Any] = {"operation": operation, "status": "started", **context}
        self.info(f"Starting operation: {operation}", extra=extra)
    
    def log_operation_complete(self, operation: str, **context: Any) -> None:
        """
        Log the successful completion of an operation.
        
        Parameters
        ----------
        operation : str
            Name of the operation
        **context
            Additional context key-value pairs
        """
        extra: Dict[str, Any] = {"operation": operation, "status": "completed", **context}
        self.info(f"Completed operation: {operation}", extra=extra)
    
    def log_operation_failed(self, operation: str, error: Exception, **context: Any) -> None:
        """
        Log a failed operation with exception details.
        
        Parameters
        ----------
        operation : str
            Name of the operation
        error : Exception
            The exception that occurred
        **context
            Additional context key-value pairs
        """
        extra: Dict[str, Any] = {
            "operation": operation,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        self.error(f"Failed operation: {operation}", extra=extra, exc_info=True)


# Factory function for creating logger instances with custom names
def create_logger(name: str) -> Logger:
    """
    Create a logging service instance with a custom name.
    
    Parameters
    ----------
    name : str
        Logger name (typically module or service name)
    
    Returns
    -------
    Logger
        Configured logging service instance
    """
    return Logger(name)
