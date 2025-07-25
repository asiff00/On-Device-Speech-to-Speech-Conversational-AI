from pathlib import Path
import os
import platform
from dotenv import load_dotenv
load_dotenv()

def init_espeak():
    """Initialize eSpeak environment variables. Must be called before any other imports."""
    system = platform.system()
    if system == "Windows":
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
        os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"


init_espeak()

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Settings class to manage application configurations."""

    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "data" / "models"
    VOICES_DIR: Path = BASE_DIR / "data" / "voices"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    RECORDINGS_DIR: Path = BASE_DIR / "recordings"
    
    # Whisper settings
    WHISPER_CPP_PATH: Optional[str] = Field(default=None, env="WHISPER_CPP_PATH", description="Optional path to whisper.cpp executable")

    ESPEAK_LIBRARY_PATH: str = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    ESPEAK_PATH: str = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

    TTS_MODEL: str = Field(..., env="TTS_MODEL")
    VOICE_NAME: str = Field(..., env="VOICE_NAME")
    SPEED: float = Field(default=1.0, env="SPEED")
    HUGGINGFACE_TOKEN: str = Field(..., env="HUGGINGFACE_TOKEN")

    LM_STUDIO_URL: str = Field(..., env="LM_STUDIO_URL")
    OLLAMA_URL: str = Field(..., env="OLLAMA_URL")
    DEFAULT_SYSTEM_PROMPT: str = Field(..., env="DEFAULT_SYSTEM_PROMPT")
    LLM_MODEL: str = Field(..., env="LLM_MODEL")
    NUM_THREADS: int = Field(default=2, env="NUM_THREADS")
    MAX_TOKENS: int = Field(default=512, env="MAX_TOKENS")
    LLM_TEMPERATURE: float = Field(default=0.7, env="LMM_TEMPERATURE")
    LLM_STREAM: bool = Field(default=False, env="LLM_STREAM")
    LLM_RETRY_DELAY: float = Field(default=0.5, env="LLM_RETRY_DELAY")
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")

    WHISPER_MODEL: str = Field(default="openai/whisper-tiny.en", env="WHISPER_MODEL")

    VAD_MODEL: str = Field(default="pyannote/segmentation-3.0", env="VAD_MODEL")
    VAD_MIN_DURATION_ON: float = Field(default=0.1, env="VAD_MIN_DURATION_ON")
    VAD_MIN_DURATION_OFF: float = Field(default=0.1, env="VAD_MIN_DURATION_OFF")

    CHUNK: int = Field(default=1024, env="CHUNK")
    FORMAT: str = Field(default="pyaudio.paFloat32", env="FORMAT")
    CHANNELS: int = Field(default=1, env="CHANNELS")
    RATE: int = Field(default=16000, env="RATE")
    OUTPUT_SAMPLE_RATE: int = Field(default=24000, env="OUTPUT_SAMPLE_RATE")
    RECORD_DURATION: int = Field(default=5, env="RECORD_DURATION")
    SILENCE_THRESHOLD: float = Field(default=0.01, env="SILENCE_THRESHOLD")
    INTERRUPTION_THRESHOLD: float = Field(default=0.02, env="INTERRUPTION_THRESHOLD")
    MAX_SILENCE_DURATION: int = Field(default=1, env="MAX_SILENCE_DURATION")
    SPEECH_CHECK_TIMEOUT: float = Field(default=0.1, env="SPEECH_CHECK_TIMEOUT")
    SPEECH_CHECK_THRESHOLD: float = Field(default=0.02, env="SPEECH_CHECK_THRESHOLD")
    ROLLING_BUFFER_TIME: float = Field(default=0.5, env="ROLLING_BUFFER_TIME")
    TARGET_SIZE: int = Field(default=15, env="TARGET_SIZE")
    FIRST_SENTENCE_SIZE: int = Field(default=3, env="FIRST_SENTENCE_SIZE")
    PLAYBACK_DELAY: float = Field(default=0.005, env="PLAYBACK_DELAY")

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.VOICES_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def configure_logging():
    """Configure logging to suppress all logs"""
    import logging
    import warnings

    warnings.filterwarnings("ignore")

    logging.getLogger().setLevel(logging.ERROR)

    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("whisper").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger("sounddevice").setLevel(logging.ERROR)
    logging.getLogger("soundfile").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.ERROR)
    logging.getLogger("fastapi").setLevel(logging.ERROR)


configure_logging()
