"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.config import settings
from app.core.logging import setup_logging
from app.core.rate_limit import limiter
from app.core.tracing import setup_langsmith
from app.db.session import create_db_and_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events, including checkpointer initialization.
    """
    # Startup
    setup_logging()
    setup_langsmith()  # Now reads from settings directly

    # Create database tables if they don't exist
    await create_db_and_tables()

    # Initialize MCP client manager
    from app.mcp.client import MCPClientManager
    from app.config import settings as app_settings

    try:
        async with MCPClientManager(app_settings.mcp_servers_config) as mcp_manager:
            # Store MCP manager in app state
            app.state.mcp_manager = mcp_manager

            # Get tool counts
            all_tools = mcp_manager.get_all_tools()
            total_tools = sum(len(tools) for tools in all_tools.values())
            enabled_servers = mcp_manager.get_enabled_servers()

            print(f"✓ MCP Manager initialized")
            print(f"  - {len(enabled_servers)} server(s) enabled: {', '.join(enabled_servers)}")
            print(f"  - {total_tools} tool(s) available from external servers")

            # Initialize PostgreSQL checkpointer for conversation memory
            # Convert asyncpg URL to psycopg format (remove +asyncpg)
            checkpointer_url = settings.database_url.replace("+asyncpg", "")

            # Use async context manager to manage connection lifecycle
            async with AsyncPostgresSaver.from_conn_string(checkpointer_url) as checkpointer:
                # Setup checkpoint tables
                await checkpointer.setup()

                # Store checkpointer in app state for access in routes
                app.state.checkpointer = checkpointer

                print(f"✓ Checkpointer initialized with PostgreSQL")
                print(
                    f"✓ Database: {checkpointer_url.split('@')[1] if '@' in checkpointer_url else 'configured'}"
                )

                yield

                # Cleanup happens automatically when exiting async contexts
                print("Shutting down checkpointer...")

            print("Shutting down MCP manager...")

    except Exception as e:
        print(f"Error initializing MCP manager: {e}")
        print("GitHub MCP initialization failed")

        # Fallback: continue without MCP manager
        app.state.mcp_manager = None

        # Still initialize checkpointer
        checkpointer_url = settings.database_url.replace("+asyncpg", "")
        async with AsyncPostgresSaver.from_conn_string(checkpointer_url) as checkpointer:
            await checkpointer.setup()
            app.state.checkpointer = checkpointer
            print(f"✓ Checkpointer initialized (fallback mode)")
            yield


# Create FastAPI app
app = FastAPI(
    title="BugTraceAgent",
    description="GitHub issue root-cause analysis agent powered by LangGraph and MCP",
    version="0.1.0",
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.middleware("http")
async def add_request_to_limiter(request: Request, call_next):
    """Add request to rate limiter context."""
    request.state.limiter = limiter
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
