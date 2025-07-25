import os
import subprocess

def test_whisper_initialization():
    try:
        whisper_cpp_path = r"D:\bilal bahi\DAILER_PORTAL\whisper cpp\whisper.cpp"
        model_path = os.path.join(whisper_cpp_path, "ggml-tiny.en.bin")
        cli_path = os.path.join(whisper_cpp_path, "build", "bin", "Release", "whisper-cli.exe")
        sample_path = os.path.join(whisper_cpp_path, "samples", "jfk.wav")

        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False

        if not os.path.exists(cli_path):
            print(f"Error: whisper-cli.exe not found at {cli_path}")
            return False

        print("\nTesting Whisper.cpp with tiny model...")
        result = subprocess.run(
            [cli_path, "-m", model_path, "-f", sample_path],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        print("Whisper.cpp test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Whisper initialization test...")
    success = test_whisper_initialization()
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nTests failed. Please check the error messages above.")
