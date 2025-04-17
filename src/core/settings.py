import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field

# Определяем базовую директорию проекта
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class OpenAISettings(BaseSettings):
    """Настройки для подключения к OpenAI API"""
    api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    model: str = Field("gpt-4o", validation_alias="OPENAI_MODEL_NAME")
    max_tokens: int = Field(4000, validation_alias="OPENAI_MAX_TOKENS")
    temperature: float = Field(0.3, validation_alias="OPENAI_TEMPERATURE")
    request_timeout: int = Field(120, validation_alias="OPENAI_REQUEST_TIMEOUT")

    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env", env_file_encoding='utf-8', extra='ignore')


class AppSettings(BaseSettings):
    """Основные настройки приложения"""
    # Директории
    input_dir: str = Field(..., validation_alias="INPUT_DIR")
    output_dir: str = Field(..., validation_alias="OUTPUT_DIR")
    prompts_dir: str = Field("src/prompt", validation_alias="PROMPTS_DIR")

    # Настройки производительности
    thread_count: int = Field(5, validation_alias="THREAD_COUNT")

    @computed_field
    def absolute_input_dir(self) -> str:
        """Возвращает абсолютный путь к директории ввода"""
        return os.path.abspath(self.input_dir)

    @computed_field
    def absolute_output_dir(self) -> str:
        """Возвращает абсолютный путь к директории вывода"""
        return os.path.abspath(self.output_dir)

    @computed_field
    def absolute_prompts_dir(self) -> str:
        """Возвращает абсолютный путь к директории с промптами"""
        return os.path.abspath(self.prompts_dir)

    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env", env_file_encoding='utf-8', extra='ignore')


class Settings(BaseSettings):
    """Общие настройки приложения"""
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    app: AppSettings = Field(default_factory=AppSettings)


    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env", env_file_encoding='utf-8', extra='ignore')


# Создаем экземпляр настроек
settings = Settings()