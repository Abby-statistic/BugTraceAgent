"""LangSmith tracing integration.

Provides observability for LangGraph agent workflows.
"""

import os


def setup_langsmith() -> None:
    """
    Setup LangSmith tracing for LangGraph workflows.
    
    LangSmith provides observability for your agent's execution.
    Get your API key at: https://smith.langchain.com/
    
    Reads configuration from app.config.settings
    """
    from app.config import settings
    
    if settings.langsmith_enabled and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        print(f"✓ LangSmith tracing enabled (project: {settings.langsmith_project})")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if not settings.langsmith_enabled:
            print("✓ LangSmith tracing disabled (LANGSMITH_ENABLED=false)")
        elif not settings.langsmith_api_key:
            print("✓ LangSmith tracing disabled (no API key)")


def is_langsmith_enabled() -> bool:
    """
    Check if LangSmith tracing is enabled.
    
    Returns:
        True if tracing is enabled, False otherwise
    """
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
