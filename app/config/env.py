import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
    API_KEY: str | None = os.getenv("API_KEY") or None
    CHROMADATABASE_HOST: str | None = os.getenv("CHROMADATABASE_HOST") or None


def get_settings() -> Settings:
    return Settings()
