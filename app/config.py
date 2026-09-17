"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"

    # LLM Configuration
    model_name: str = "deepseek-flash"
    model_temperature: float = 0.1
    model_max_tokens: int = 3000
    # GitHub MCP Configuration
    github_personal_access_token: str = ""

    # MCP Configuration
    mcp_transport: str = "stdio"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # CORS Configuration
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Database Configuration
    database_url: str = "postgresql+asyncpg://agent:agent_password@localhost:5432/agent_db"
    database_echo: bool = False

    # Authentication Configuration
    api_keys: str = ""
    auth_enabled: bool = False

    # Rate Limiting Configuration
    rate_limit_per_minute: int = 100

    # LangSmith Configuration (Optional)
    langsmith_api_key: str = ""
    langsmith_project: str = "bugtrace-agent"
    langsmith_enabled: bool = False

    # Logging Configuration
    log_level: str = "INFO"
    json_logs: bool = False

    # MCP Server Configuration
    mcp_servers_config: str = "mcp_servers.json"
    mcp_timeout: int = 30  # seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),  # Fix Pydantic warning
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def api_keys_list(self) -> list[str]:
        """Parse API keys from comma-separated string."""
        if not self.api_keys:
            return []
        return [key.strip() for key in self.api_keys.split(",")]


# Global settings instance
settings = Settings()
