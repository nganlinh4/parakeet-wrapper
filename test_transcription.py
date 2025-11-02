import requests
import os

def test_transcription_api():
    """Test the transcription API with the example audio file."""
    audio_file_path = "data/example-yt_saTD1u8PorI.mp3"

    if not os.path.exists(audio_file_path):
        print(f"Error: Test audio file not found at {audio_file_path}")
        return

    try:
        # Make API request
        with open(audio_file_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_file_path), f, 'audio/mpeg')}
            response = requests.post('http://localhost:8001/transcribe', files=files)

        if response.status_code == 200:
            result = response.json()

            # Print basic info
            print(f"Transcription successful!")
            print(f"Duration: {result['duration']:.2f} seconds")
            print(f"Number of segments: {len(result['segments'])}")

            # Save SRT content
            with open("transcription.srt", "w", encoding="utf-8") as f:
                f.write(result['srt_content'])

            print("SRT file saved as 'transcription.srt'")
            print("\nFirst few segments:")
            for i, segment in enumerate(result['segments'][:3]):
                print(f"  {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s: {segment['segment'][:50]}...")

        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_transcription_api()