import os
import subprocess
import soundfile as sf
import numpy as np

def transcribe_audio_local(audio_data, sample_rate=16000):
    """Transcribes audio using the local whisper.cpp executable.
    
    Args:
        audio_data (np.ndarray): Audio data to transcribe.
        sample_rate (int): Sample rate of the audio.
        
    Returns:
        str: Transcribed text.
    """
    # Define the paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    whisper_cpp_dir = os.path.join(os.path.dirname(base_dir), "whisper.cpp")
    whisper_bin_path = os.path.join(whisper_cpp_dir, "build", "bin", "Release", "whisper-cli.exe")
    tiny_model_path = os.path.join(whisper_cpp_dir, "models", "ggml-tiny.en.bin")
    temp_wav_path = os.path.join(base_dir, "output", "temp_audio_for_transcription.wav")
    
    try:
        # Save the audio to a temporary WAV file
        os.makedirs(os.path.dirname(temp_wav_path), exist_ok=True)
        sf.write(temp_wav_path, audio_data, sample_rate, 'PCM_16')
        
        # Prepare the command
        cmd = [
            whisper_bin_path,
            "-m", tiny_model_path,
            "-f", temp_wav_path,
            "-otxt",
            "-pp"  # Enable post-processing (language detection, special tokens, etc.)
        ]
        
        # Run whisper.cpp
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            text = result.stdout.strip()
            
            # If no output in stdout, try reading the .txt file
            if not text:
                txt_path = os.path.splitext(temp_wav_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
            
            return text
            
        except subprocess.CalledProcessError as e:
            print(f"Error running whisper.cpp: {e.stderr}")
            return ""
            
        finally:
            # Clean up
            try:
                os.remove(temp_wav_path)
                txt_path = os.path.splitext(temp_wav_path)[0] + ".txt"
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            except:
                pass
                
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return ""
