from onnx_asr import load_model
import gc
from pathlib import Path
from pydub import AudioSegment
import numpy as np
import os
import csv
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import shutil

MODEL_NAME="istupakov/parakeet-tdt-0.6b-v3-onnx"

model = load_model(MODEL_NAME)

app = FastAPI(title="Parakeet Speech Transcription API", version="1.0.0")

def get_audio_segment(audio_path, start_second, end_second):
    if not audio_path or not Path(audio_path).exists():
        print(f"Warning: Audio path '{audio_path}' not found or invalid for clipping.")
        return None
    try:
        start_ms = int(start_second * 1000)
        end_ms = int(end_second * 1000)

        start_ms = max(0, start_ms)
        if end_ms <= start_ms:
            print(f"Warning: End time ({end_second}s) is not after start time ({start_second}s). Adjusting end time.")
            end_ms = start_ms + 100

        audio = AudioSegment.from_file(audio_path)
        clipped_audio = audio[start_ms:end_ms]

        samples = np.array(clipped_audio.get_array_of_samples())
        if clipped_audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1).astype(samples.dtype)

        frame_rate = clipped_audio.frame_rate
        if frame_rate <= 0:
             print(f"Warning: Invalid frame rate ({frame_rate}) detected for clipped audio.")
             frame_rate = audio.frame_rate

        if samples.size == 0:
             print(f"Warning: Clipped audio resulted in empty samples array ({start_second}s to {end_second}s).")
             return None

        return (frame_rate, samples)
    except FileNotFoundError:
        print(f"Error: Audio file not found at path: {audio_path}")
        return None
    except Exception as e:
        print(f"Error clipping audio {audio_path} from {start_second}s to {end_second}s: {e}")
        return None

def format_srt_time(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm using datetime.timedelta"""
    sanitized_total_seconds = max(0.0, seconds)
    delta = datetime.timedelta(seconds=sanitized_total_seconds)
    total_int_seconds = int(delta.total_seconds())

    hours = total_int_seconds // 3600
    remainder_seconds_after_hours = total_int_seconds % 3600
    minutes = remainder_seconds_after_hours // 60
    seconds_part = remainder_seconds_after_hours % 60
    milliseconds = delta.microseconds // 1000

    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

def generate_srt_content(segment_timestamps: list) -> str:
    """Generates SRT formatted string from segment timestamps."""
    srt_content = []
    for i, ts in enumerate(segment_timestamps):
        start_time = format_srt_time(ts['start'])
        end_time = format_srt_time(ts['end'])
        text = ts['segment']
        srt_content.append(str(i + 1))
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")
    return "\n".join(srt_content)

def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio file and return results as JSON."""
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=400, detail="No audio file path provided or file does not exist.")

    processed_audio_path = None
    original_path_name = Path(audio_path).name
    audio_name = Path(audio_path).stem

    try:
        try:
            print(f"Loading audio: {original_path_name}")
            audio = AudioSegment.from_file(audio_path)
            duration_sec = audio.duration_seconds
        except Exception as load_e:
            raise HTTPException(status_code=400, detail=f"Failed to load audio file {original_path_name}: {load_e}")

        resampled = False
        mono = False

        target_sr = 16000
        if audio.frame_rate != target_sr:
            try:
                audio = audio.set_frame_rate(target_sr)
                resampled = True
            except Exception as resample_e:
                raise HTTPException(status_code=400, detail=f"Failed to resample audio: {resample_e}")

        if audio.channels == 2:
            try:
                audio = audio.set_channels(1)
                mono = True
            except Exception as mono_e:
                raise HTTPException(status_code=400, detail=f"Failed to convert audio to mono: {mono_e}")
        elif audio.channels > 2:
            raise HTTPException(status_code=400, detail=f"Audio has {audio.channels} channels. Only mono (1) or stereo (2) supported.")

        if resampled or mono:
            try:
                with tempfile.NamedTemporaryFile(suffix='_resampled.wav', delete=False) as temp_file:
                    processed_audio_path = Path(temp_file.name)
                audio.export(processed_audio_path, format="wav")
                transcribe_path = str(processed_audio_path)
                info_path_name = f"{original_path_name} (processed)"
            except Exception as export_e:
                if processed_audio_path and processed_audio_path.exists():
                    processed_audio_path.unlink()
                raise HTTPException(status_code=500, detail=f"Failed to export processed audio: {export_e}")
        else:
            transcribe_path = audio_path
            info_path_name = original_path_name

        try:
            print(f"Transcribing {info_path_name}...")

            result = model.recognize(transcribe_path)
            transcription = result
            segment_timestamps = [{'start': 0.0, 'end': duration_sec, 'segment': transcription}]

            # Generate CSV content
            csv_content = []
            csv_headers = ["Start (s)", "End (s)", "Segment"]
            csv_content.append(csv_headers)
            for ts in segment_timestamps:
                csv_content.append([f"{ts['start']:.2f}", f"{ts['end']:.2f}", ts['segment']])

            # Generate SRT content
            srt_content = generate_srt_content(segment_timestamps)

            print("Transcription complete.")
            return {
                "transcription": transcription,
                "segments": segment_timestamps,
                "csv_data": csv_content,
                "srt_content": srt_content,
                "duration": duration_sec
            }

        except FileNotFoundError:
            error_msg = f"Audio file for transcription not found: {Path(transcribe_path).name}."
            print(f"Error: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)

        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            print(f"Error during transcription processing: {e}")
            raise HTTPException(status_code=500, detail=error_msg)
        finally:
            # Model cleanup
            try:
                gc.collect()
            except Exception as cleanup_e:
                print(f"Error during model cleanup: {cleanup_e}")

    finally:
        if processed_audio_path and processed_audio_path.exists():
            try:
                processed_audio_path.unlink()
                print(f"Temporary audio file {processed_audio_path} removed.")
            except Exception as e:
                print(f"Error removing temporary audio file {processed_audio_path}: {e}")


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """Transcribe uploaded audio file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        result = transcribe_audio(temp_path)
        return JSONResponse(content=result)
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

@app.get("/")
async def root():
    return {"message": "Parakeet Speech Transcription API", "version": "1.0.0"}

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)