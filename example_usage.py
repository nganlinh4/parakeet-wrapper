import requests
import os
import argparse
import sys

# --- Configuration ---
API_URL = "http://localhost:8001/transcribe"
DEFAULT_AUDIO_PATH = "data/example-yt_saTD1u8PorI.mp3"

def transcribe_audio_file(audio_file_path: str, strategy: str, max_chars: int):
    """
    Sends an audio file to the transcription API with segmentation parameters.
    """
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found at '{audio_file_path}'")
        sys.exit(1)

    file_name = os.path.basename(audio_file_path)
    output_srt_path = f"{os.path.splitext(file_name)[0]}_{strategy}.srt"

    print(f"🚀 Sending '{file_name}' to the API with strategy='{strategy}' and max_chars={max_chars}...")
    print("   (This may take a moment for large files)")

    try:
        with open(audio_file_path, 'rb') as f:
            # We send parameters as a 'data' payload alongside the 'files'
            form_data = {
                'segment_strategy': strategy,
                'max_chars': str(max_chars) # Form data should be strings
            }
            files = {'file': (file_name, f, 'audio/mpeg')}
            
            response = requests.post(API_URL, data=form_data, files=files, timeout=300)

        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        print(f"   Please ensure the FastAPI server is running and accessible at {API_URL}")
        sys.exit(1)
        
    # --- (The rest of the function for processing the response is the same as before) ---
    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        print("❌ Error: Failed to decode JSON from API response.")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Text: {response.text}")
        sys.exit(1)

    if 'transcription' in result and 'segments' in result:
        print("\n✅ Transcription successful!")
        duration = result.get('duration_seconds', 0.0)
        print(f"   Audio Duration: {duration:.2f} seconds")
        print(f"   Number of segments: {len(result['segments'])}")
        
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(result['srt_content'])
        print(f"   💾 SRT file saved as '{output_srt_path}'")

        print("\n📝 First 3 Segments:")
        for i, segment in enumerate(result['segments'][:3]):
            start, end, text = segment.get('start', 0), segment.get('end', 0), segment.get('segment', '')
            print(f"  {i+1}: [{start:0>7.2f}s -> {end:0>7.2f}s] {text}")
    else:
        print("❌ API returned a successful status but the response is malformed.")
        print(f"   Response: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Parakeet Speech Transcription API with tunable segmentation."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=DEFAULT_AUDIO_PATH,
        help=f"Path to the audio file to transcribe. Defaults to '{DEFAULT_AUDIO_PATH}'"
    )
    parser.add_argument(
        "-s", "--strategy",
        choices=['char', 'sentence'],
        default='sentence',
        help="The segmentation strategy to use. 'char' is word-safe character-limited. 'sentence' splits by sentence."
    )
    parser.add_argument(
        "-c", "--max_chars",
        type=int,
        default=60,
        help="The maximum characters per segment for the 'char' strategy."
    )
    args = parser.parse_args()
    
    transcribe_audio_file(args.audio_file, args.strategy, args.max_chars)