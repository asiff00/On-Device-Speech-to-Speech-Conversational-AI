import keyboard
import traceback
import time 
import requests
import subprocess
import numpy as np
import wave
import os
from src.utils.config import settings as base_settings
from src.utils.config_local import LocalSettings
from src.utils.speech_local import (
    init_vad_pipeline,
    detect_speech_segments,
    record_continuous_audio,
    check_for_speech,
    play_audio_with_interrupt,
)
from src.utils import (
    VoiceGenerator,
    get_ai_response,
)

settings = LocalSettings()
from src.utils.audio_queue import AudioGenerationQueue
from src.utils.llm import parse_stream_chunk
import threading
from src.utils.text_chunker import TextChunker

settings.setup_directories()
timing_info = {
    "vad_start": None,
    "transcription_start": None,
    "llm_first_token": None,
    "audio_queued": None,
    "first_audio_play": None,
    "playback_start": None,
    "end": None,
    "transcription_duration": None,
}

WHISPER_MODEL_PATH = os.path.join(settings.WHISPER_CPP_PATH, settings.WHISPER_MODEL)
WHISPER_CLI_PATH = os.path.join(settings.WHISPER_CPP_PATH, "build", "bin", "Release", "whisper-cli.exe")

def transcribe_audio_local(audio_data, sample_rate=16000):
    """Transcribe audio using local whisper.cpp model."""
    try:
        # Save the audio data to a temporary WAV file
        temp_wav = "temp_speech.wav"
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        # Run whisper-cli on the temporary file
        result = subprocess.run(
            [WHISPER_CLI_PATH, "-m", WHISPER_MODEL_PATH, "-f", temp_wav],
            capture_output=True,
            text=True
        )

        # Clean up the temporary file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        # Extract the transcription from the output
        # Whisper.cpp outputs with timestamps like [00:00:00.000 --> 00:00:10.000] Text here
        # We'll extract just the text part
        lines = result.stdout.split('\n')
        transcription = []
        for line in lines:
            if ']' in line:
                text = line.split(']')[-1].strip()
                if text:
                    transcription.append(text)

        return ' '.join(transcription)

    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return ""

def process_input(
    session: requests.Session,
    user_input: str,
    messages: list,
    generator: VoiceGenerator,
    speed: float,
) -> tuple[bool, None]:
    """Processes user input, generates a response, and handles audio output."""
    global timing_info
    timing_info = {k: None for k in timing_info}
    timing_info["vad_start"] = time.perf_counter()

    messages.append({"role": "user", "content": user_input})
    print("\nThinking...")
    try:
        response_stream = get_ai_response(
            session=session,
            messages=messages,
            llm_model=settings.LLM_MODEL,
            llm_url=settings.OLLAMA_URL,
            max_tokens=settings.MAX_TOKENS,
            stream=True,
        )

        if not response_stream:
            print("Failed to get AI response stream.")
            return False, None

        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()
        chunker = TextChunker()
        complete_response = []

        playback_thread = threading.Thread(
            target=lambda: audio_playback_worker(audio_queue)
        )
        playback_thread.daemon = True
        playback_thread.start()

        for chunk in response_stream:
            data = parse_stream_chunk(chunk)
            if not data or "choices" not in data:
                continue

            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                content = choice["delta"]["content"]
                if content:
                    if not timing_info["llm_first_token"]:
                        timing_info["llm_first_token"] = time.perf_counter()
                    print(content, end="", flush=True)
                    chunker.current_text.append(content)

                    text = "".join(chunker.current_text)
                    if chunker.should_process(text):
                        if not timing_info["audio_queued"]:
                            timing_info["audio_queued"] = time.perf_counter()
                        remaining = chunker.process(text, audio_queue)
                        chunker.current_text = [remaining]
                        complete_response.append(text[: len(text) - len(remaining)])

            if choice.get("finish_reason") == "stop":
                final_text = "".join(chunker.current_text).strip()
                if final_text:
                    chunker.process(final_text, audio_queue)
                    complete_response.append(final_text)
                break

        messages.append({"role": "assistant", "content": " ".join(complete_response)})
        print()

        time.sleep(0.1)
        audio_queue.stop()
        playback_thread.join()

        def playback_wrapper():
            timing_info["playback_start"] = time.perf_counter()
            result = audio_playback_worker(audio_queue)
            return result

        playback_thread = threading.Thread(target=playback_wrapper)

        timing_info["end"] = time.perf_counter()
        print_timing_chart(timing_info)
        return False, None

    except Exception as e:
        print(f"\nError during streaming: {str(e)}")
        if "audio_queue" in locals():
            audio_queue.stop()
        return False, None

def audio_playback_worker(audio_queue) -> tuple[bool, None]:
    """Manages audio playback in a separate thread, handling interruptions."""
    global timing_info
    was_interrupted = False
    interrupt_audio = None

    try:
        while True:
            speech_detected, audio_data = check_for_speech()
            if speech_detected:
                was_interrupted = True
                interrupt_audio = audio_data
                break

            audio_data, _ = audio_queue.get_next_audio()
            if audio_data is not None:
                if not timing_info["first_audio_play"]:
                    timing_info["first_audio_play"] = time.perf_counter()

                was_interrupted, interrupt_data = play_audio_with_interrupt(audio_data)
                if was_interrupted:
                    interrupt_audio = interrupt_data
                    break
            else:
                time.sleep(settings.PLAYBACK_DELAY)

            if (
                not audio_queue.is_running
                and audio_queue.sentence_queue.empty()
                and audio_queue.audio_queue.empty()
            ):
                break

    except Exception as e:
        print(f"Error in audio playback: {str(e)}")

    return was_interrupted, interrupt_audio

def main():
    """Main function to run the voice chat bot."""
    with requests.Session() as session:
        try:
            session = requests.Session()
            generator = VoiceGenerator(settings.MODELS_DIR, settings.VOICES_DIR)
            messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
            
            # Test whisper.cpp
            print("\nTesting Whisper.cpp...")
            test_result = subprocess.run(
                [WHISPER_CLI_PATH, "-m", WHISPER_MODEL_PATH, "-f", os.path.join(WHISPER_CPP_PATH, "samples", "jfk.wav")],
                capture_output=True,
                text=True
            )
            if test_result.returncode != 0:
                print("Error: Failed to initialize Whisper.cpp")
                print(test_result.stderr)
                return
            print("Whisper.cpp initialized successfully!")

            print("\nInitializing Voice Activity Detection...")
            vad_pipeline = init_vad_pipeline(settings.HUGGINGFACE_TOKEN)
            print("\n=== Voice Chat Bot Initializing ===")
            print("Device being used:", generator.device)
            print("\nInitializing voice generator...")
            result = generator.initialize(settings.TTS_MODEL, settings.VOICE_NAME)
            print(result)
            speed = settings.SPEED

            try:
                print("\nWarming up the LLM model...")
                health = session.get("http://localhost:11434", timeout=3)
                if health.status_code != 200:
                    print("Ollama not running! Start it first.")
                    return
                response_stream = get_ai_response(
                    session=session,
                    messages=[
                        {"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": "Hi!"},
                    ],
                    llm_model=settings.LLM_MODEL,
                    llm_url=settings.OLLAMA_URL,
                    max_tokens=settings.MAX_TOKENS,
                    stream=False,
                )
                if not response_stream:
                    print("Failed to initialize the AI model!")
                    return
            except requests.RequestException as e:
                print(f"Warmup failed: {str(e)}")

            print("\n\n=== Voice Chat Bot Ready ===")
            print("The bot is now listening for speech.")
            print("Just start speaking, and I'll respond automatically!")
            print("You can interrupt me anytime by starting to speak.")
            while True:
                try:
                    if keyboard.is_pressed("enter"):
                        user_input = input("\nYou (text): ").strip()
                        if user_input.lower() == "quit":
                            print("Goodbye!")
                            break

                    audio_data = record_continuous_audio()
                    if audio_data is not None:
                        speech_segments = detect_speech_segments(
                            vad_pipeline, audio_data
                        )

                        if speech_segments is not None:
                            print("\nTranscribing detected speech...")
                            timing_info["transcription_start"] = time.perf_counter()

                            user_input = transcribe_audio_local(speech_segments)

                            timing_info["transcription_duration"] = (
                                time.perf_counter() - timing_info["transcription_start"]
                            )
                            if user_input.strip():
                                print(f"You (voice): {user_input}")
                                was_interrupted, speech_data = process_input(
                                    session, user_input, messages, generator, speed
                                )
                                if was_interrupted and speech_data is not None:
                                    speech_segments = detect_speech_segments(
                                        vad_pipeline, speech_data
                                    )
                                    if speech_segments is not None:
                                        print("\nTranscribing interrupted speech...")
                                        user_input = transcribe_audio_local(speech_segments)
                                        if user_input.strip():
                                            print(f"You (voice): {user_input}")
                                            process_input(
                                                session,
                                                user_input,
                                                messages,
                                                generator,
                                                speed,
                                            )
                        else:
                            print("No clear speech detected, please try again.")

                    if session is not None:
                        session.headers.update({"Connection": "keep-alive"})
                        if hasattr(session, "connection_pool"):
                            session.connection_pool.clear()

                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()

def print_timing_chart(metrics):
    """Prints timing chart from global metrics"""
    base_time = metrics["vad_start"]
    events = [
        ("User stopped speaking", metrics["vad_start"]),
        ("VAD started", metrics["vad_start"]),
        ("Transcription started", metrics["transcription_start"]),
        ("LLM first token", metrics["llm_first_token"]),
        ("Audio queued", metrics["audio_queued"]),
        ("First audio played", metrics["first_audio_play"]),
        ("Playback started", metrics["playback_start"]),
        ("End-to-end response", metrics["end"]),
    ]

    print("\nTiming Chart:")
    print(f"{'Event':<25} | {'Time (s)':>9} | {'Δ+':>6}")
    print("-" * 45)

    prev_time = base_time
    for name, t in events:
        if t is None:
            continue
        elapsed = t - base_time
        delta = t - prev_time
        print(f"{name:<25} | {elapsed:9.2f} | {delta:6.2f}")
        prev_time = t

if __name__ == "__main__":
    main()
