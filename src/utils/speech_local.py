import pyaudio
import numpy as np
import torch
from torch.nn.functional import pad
import time
from queue import Queue
import sounddevice as sd
from .config import settings

CHUNK = settings.CHUNK
FORMAT = pyaudio.paFloat32
CHANNELS = settings.CHANNELS
RATE = settings.RATE
SILENCE_THRESHOLD = settings.SILENCE_THRESHOLD
SPEECH_CHECK_THRESHOLD = settings.SPEECH_CHECK_THRESHOLD
MAX_SILENCE_DURATION = settings.MAX_SILENCE_DURATION


def init_vad_pipeline(hf_token):
    """Initializes the Voice Activity Detection pipeline.

    Args:
        hf_token (str): Hugging Face API token.

    Returns:
        pyannote.audio.pipelines.VoiceActivityDetection: VAD pipeline.
    """
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection
    import os
    
    # Ensure the token is set in the environment
    os.environ["HF_TOKEN"] = hf_token
    
    try:
        # Try loading the model with authentication
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=hf_token
        )
        
        pipeline = VoiceActivityDetection(segmentation=model)

        HYPER_PARAMETERS = {
            "min_duration_on": settings.VAD_MIN_DURATION_ON,
            "min_duration_off": settings.VAD_MIN_DURATION_OFF,
        }
        pipeline.instantiate(HYPER_PARAMETERS)
        
        return pipeline
        
    except Exception as e:
        print(f"\nError initializing VAD pipeline: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Accepted the user agreement at: https://huggingface.co/pyannote/segmentation-3.0")
        print("2. Created an access token at: https://huggingface.co/settings/tokens")
        print("3. Added your token to the .env file as: HUGGINGFACE_TOKEN=your_token_here")
        return None


def detect_speech_segments(pipeline, audio_data, sample_rate=None):
    """Detects speech segments in audio using pyannote VAD.

    Args:
        pipeline (pyannote.audio.pipelines.VoiceActivityDetection): VAD pipeline.
        audio_data (np.ndarray or torch.Tensor): Audio data.
        sample_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        torch.Tensor or None: Concatenated speech segments as a torch tensor, or None if no speech is detected.
    """
    if sample_rate is None:
        sample_rate = settings.RATE

    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(1, -1)

    if not isinstance(audio_data, torch.Tensor):
        audio_data = torch.from_numpy(audio_data)

    if audio_data.shape[1] < sample_rate:
        padding_size = sample_rate - audio_data.shape[1]
        audio_data = pad(audio_data, (0, padding_size))

    try:
        vad = pipeline({"waveform": audio_data, "sample_rate": sample_rate})

        speech_segments = []
        for speech in vad.get_timeline().support():
            start_sample = int(speech.start * sample_rate)
            end_sample = int(speech.end * sample_rate)
            if start_sample < audio_data.shape[1]:
                end_sample = min(end_sample, audio_data.shape[1])
                segment = audio_data[0, start_sample:end_sample]
                speech_segments.append(segment)

        if speech_segments:
            return torch.cat(speech_segments)
        return None
    
    except Exception as e:
        print(f"\nError detecting speech segments: {str(e)}")
        return None


def record_continuous_audio():
    """Continuously monitors audio and detects speech segments.

    Returns:
        np.ndarray or None: Recorded audio data as a numpy array, or None if no speech is detected.
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("\nListening... (Press Ctrl+C to stop)")
    frames = []
    buffer_frames = []
    buffer_size = int(RATE * 0.5 / CHUNK)
    silence_frames = 0
    max_silence_frames = int(RATE / CHUNK * MAX_SILENCE_DURATION)
    recording = False

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            buffer_frames.append(audio_chunk)
            if len(buffer_frames) > buffer_size:
                buffer_frames.pop(0)

            audio_level = np.abs(np.concatenate(buffer_frames)).mean()

            if audio_level > SILENCE_THRESHOLD:
                if not recording:
                    print("\nPotential speech detected...")
                    recording = True
                    frames.extend(buffer_frames)
                frames.append(audio_chunk)
                silence_frames = 0
            elif recording:
                frames.append(audio_chunk)
                silence_frames += 1

                if silence_frames >= max_silence_frames:
                    print("Processing speech segment...")
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if frames:
        return np.concatenate(frames)
    return None


def check_for_speech(timeout=0.1):
    """Checks if speech is detected in a non-blocking way.

    Args:
        timeout (float, optional): Duration to check for speech in seconds. Defaults to 0.1.

    Returns:
        tuple: A tuple containing a boolean indicating if speech was detected and the audio data.
    """
    p = pyaudio.PyAudio()

    frames = []
    is_speech = False

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        for _ in range(int(RATE * timeout / CHUNK)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)

            audio_level = np.abs(audio_chunk).mean()
            if audio_level > SPEECH_CHECK_THRESHOLD:
                is_speech = True
                break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if is_speech and frames:
        return True, np.concatenate(frames)
    return False, None


def play_audio_with_interrupt(audio_data, sample_rate=24000):
    """Plays audio while monitoring for speech interruption.

    Args:
        audio_data (np.ndarray): Audio data to play.
        sample_rate (int, optional): Sample rate for playback. Defaults to 24000.

    Returns:
        tuple: A tuple containing a boolean indicating if playback was interrupted and audio data.
    """
    interrupt_queue = Queue()

    def input_callback(indata, frames, time, status):
        if status:
            print(f"Input status: {status}")
            return

        audio_level = np.abs(indata[:, 0]).mean()
        if audio_level > settings.INTERRUPTION_THRESHOLD:
            interrupt_queue.put((True, indata.copy()))

    def output_callback(outdata, frames, time, status):
        if status:
            print(f"Output status: {status}")
            return

        if not interrupt_queue.empty():
            raise sd.CallbackStop()

        remaining = len(audio_data) - output_callback.position
        if remaining == 0:
            raise sd.CallbackStop()
        valid_frames = min(remaining, frames)
        outdata[:valid_frames, 0] = audio_data[
            output_callback.position : output_callback.position + valid_frames
        ]
        if valid_frames < frames:
            outdata[valid_frames:] = 0
        output_callback.position += valid_frames

    output_callback.position = 0

    try:
        with sd.InputStream(
            channels=1, callback=input_callback, samplerate=settings.RATE
        ):
            with sd.OutputStream(
                channels=1, callback=output_callback, samplerate=sample_rate
            ):
                while output_callback.position < len(audio_data):
                    sd.sleep(100)
                    if not interrupt_queue.empty():
                        interrupted, interrupt_audio = interrupt_queue.get()
                        return interrupted, interrupt_audio
        return False, None
    except sd.CallbackStop:
        if not interrupt_queue.empty():
            interrupted, interrupt_audio = interrupt_queue.get()
            return interrupted, interrupt_audio
        return True, None
    except Exception as e:
        print(f"Error during playback: {str(e)}")
        return False, None
