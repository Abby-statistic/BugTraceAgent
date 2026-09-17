"""Rate limiting configuration using slowapi.

Provides per-API-key rate limiting to prevent abuse.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings


def get_api_key_identifier(request):
    """
    Get identifier for rate limiting (API key or IP address).
    
    Args:
        request: FastAPI request object
        
    Returns:
        Identifier string for rate limiting
    """
    # Try to get API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    
    # Fall back to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_api_key_identifier,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"],
)
