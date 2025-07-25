from pathlib import Path
import os
import platform
from dotenv import load_dotenv
load_dotenv()

def init_espeak():
    """Initialize eSpeak environment variables."""
    system = platform.system()
    if system == "Windows":
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
        os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

init_espeak()

from pydantic_settings import BaseSettings
from pydantic import Field

from .config import Settings

class LocalSettings(Settings):
    """Settings class for local processing configurations. 
    Inherits from base Settings and overrides with local-specific values."""
    
    # Whisper.cpp specific settings
    WHISPER_CPP_PATH: str = Field("", env="WHISPER_CPP_PATH", description="Path to whisper.cpp executable")
    WHISPER_MODEL: str = Field(default="ggml-tiny.en.bin", env="WHISPER_MODEL")
    
    # Audio Processing Settings
    SILENCE_THRESHOLD: float = Field(default=0.03, env="SILENCE_THRESHOLD", description="Higher threshold for local processing")
    INTERRUPTION_THRESHOLD: float = Field(default=0.04, env="INTERRUPTION_THRESHOLD", description="Higher threshold for local processing")
    SPEECH_CHECK_THRESHOLD: float = Field(default=0.05, env="SPEECH_CHECK_THRESHOLD", description="Higher threshold for local processing")
    PLAYBACK_DELAY: float = Field(default=0.1, env="PLAYBACK_DELAY", description="Longer delay for local processing")
    
    # API Settings
    OLLAMA_URL: str = Field(default="http://localhost:11434/api/chat", env="OLLAMA_URL", description="Local Ollama server URL")
    
    class Config:
        env_file = ".env.local"  # Use a separate local environment file
        env_file_encoding = "utf-8"

settings = LocalSettings()

def configure_logging():
    """Configure logging to suppress unnecessary output"""
    import logging
    import warnings
    
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    
    # Suppress specific loggers
    for logger_name in [
        "urllib3", "PIL", "matplotlib", "torch", "tensorflow",
        "whisper", "transformers", "pyannote", "sounddevice",
        "soundfile", "uvicorn", "fastapi"
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

configure_logging()
