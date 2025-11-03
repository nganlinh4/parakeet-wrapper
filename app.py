import os
import gc
import logging
import datetime
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from onnx_asr import load_model
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

# --- Configuration and Logging Setup ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("ASR_MODEL_NAME", "istupakov/parakeet-tdt-0.6b-v3-onnx")
app_state = {}

# --- FastAPI Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading ASR model: {MODEL_NAME}...")
    try:
        model = load_model(MODEL_NAME)
        app_state["asr_model"] = model.with_timestamps()
        logger.info("ASR model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}", exc_info=True)
        app_state["asr_model"] = None
    
    yield

    logger.info("Cleaning up resources...")
    app_state.clear()
    gc.collect()
    logger.info("Shutdown complete.")

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Parakeet Speech Transcription API",
    version="1.0.0",
    lifespan=lifespan
)

# --- Helper Functions ---

def format_srt_time(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm."""
    if seconds < 0: seconds = 0.0
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def generate_srt_content(segment_timestamps: list) -> str:
    """Generates an SRT formatted string from segment data."""
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

def _group_tokens_into_words(tokens: list, timestamps: list) -> list:
    """
    Stage 1: Groups subword tokens into whole words with their start times.
    Most tokenizers prefix tokens that start a new word with a space-like character.
    """
    words = []
    current_word_tokens = []
    current_word_start_time = None

    if not tokens:
        return []
        
    for i, token in enumerate(tokens):
        if current_word_start_time is None:
            current_word_start_time = timestamps[i]

        # Heuristic: if a token starts with a space, it's a new word.
        if token.startswith(" "):
            # Finalize the previous word
            if current_word_tokens:
                word_text = "".join(current_word_tokens).strip()
                words.append({"text": word_text, "start": current_word_start_time})
            
            # Start a new word
            current_word_tokens = [token]
            current_word_start_time = timestamps[i]
        else:
            # Continue the current word
            current_word_tokens.append(token)
    
    # Add the very last word
    if current_word_tokens:
        word_text = "".join(current_word_tokens).strip()
        words.append({"text": word_text, "start": current_word_start_time})
        
    return words


def process_timestamped_result(timestamped_result, audio_duration_sec: float, strategy: str = 'char', max_chars: int = 60) -> list:
    """
    Stage 2: Groups words into subtitle segments based on the chosen strategy.
    This function now operates on whole words, preventing any splits.
    """
    timestamps = timestamped_result.timestamps
    tokens = timestamped_result.tokens

    if not timestamps or not tokens:
        return []

    # Stage 1: Group tokens into words
    words = _group_tokens_into_words(tokens, timestamps)
    if not words:
        return []

    # Stage 2: Group words into segments
    segments = []
    current_segment_words = []
    current_segment_start = words[0]['start']

    for i, word in enumerate(words):
        current_segment_words.append(word['text'])
        text = " ".join(current_segment_words)
        
        end_segment = False
        is_last_word = (i == len(words) - 1)

        if strategy == 'char':
            # Look ahead to see if the next word would push it over the limit
            next_word_len = len(words[i+1]['text']) if not is_last_word else 0
            if (len(text) + next_word_len + 1 > max_chars) or is_last_word:
                end_segment = True
        elif strategy == 'sentence':
            if word['text'].strip().endswith(('.', '?', '!')) or is_last_word:
                end_segment = True
        
        if end_segment:
            segment_end = words[i+1]['start'] if not is_last_word else audio_duration_sec
            segments.append({
                'start': current_segment_start,
                'end': segment_end,
                'segment': text
            })
            
            if not is_last_word:
                current_segment_start = words[i+1]['start']
                current_segment_words = []
                
    return segments

# --- Core Transcription Logic ---

def transcribe_audio(audio_path: str, model, strategy: str, max_chars: int) -> dict:
    # (This function remains unchanged from the previous version)
    if not model:
        raise HTTPException(status_code=503, detail="ASR model is not available.")
    
    original_path = Path(audio_path)
    if not original_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found at path: {audio_path}")

    processed_audio_path = None
    try:
        logger.info(f"Loading audio file: {original_path.name}")
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds
    except Exception as e:
        logger.error(f"Failed to load audio file {original_path.name}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not process audio file: {e}")

    target_sr = 16000
    needs_processing = (audio.frame_rate != target_sr) or (audio.channels != 1)
    
    if audio.channels > 2:
        raise HTTPException(status_code=400, detail=f"Audio has {audio.channels} channels. Only mono or stereo supported.")
    
    if needs_processing:
        logger.info("Audio requires pre-processing (resampling/mono conversion).")
        if audio.frame_rate != target_sr: audio = audio.set_frame_rate(target_sr)
        if audio.channels != 1: audio = audio.set_channels(1)
        
        with tempfile.NamedTemporaryFile(suffix='_processed.wav', delete=False) as temp_f:
            processed_audio_path = Path(temp_f.name)
        audio.export(processed_audio_path, format="wav")
        transcribe_path = str(processed_audio_path)
    else:
        transcribe_path = audio_path

    try:
        logger.info(f"Transcribing {Path(transcribe_path).name} ({duration_sec:.2f} seconds)...")
        result = model.recognize(transcribe_path)
        
        full_transcription = result.text
        segment_timestamps = process_timestamped_result(result, duration_sec, strategy, max_chars)

        csv_content = [["Start (s)", "End (s)", "Segment"]] + [
            [f"{ts['start']:.2f}", f"{ts['end']:.2f}", ts['segment']] for ts in segment_timestamps
        ]
        srt_content = generate_srt_content(segment_timestamps)

        logger.info("Transcription complete.")
        return {
            "transcription": full_transcription,
            "segments": segment_timestamps,
            "csv_data": csv_content,
            "srt_content": srt_content,
            "duration_seconds": duration_sec
        }
    except Exception as e:
        logger.error(f"Transcription failed for {Path(transcribe_path).name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {e}")
    finally:
        if processed_audio_path and processed_audio_path.exists():
            try:
                processed_audio_path.unlink()
            except Exception as e:
                logger.error(f"Error removing temporary file {processed_audio_path}: {e}")

# --- API Endpoints ---
# (These functions also remain unchanged from the previous version)

@app.get("/")
async def root():
    return {"message": "Parakeet Speech Transcription API", "version": "1.0.0"}

@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    segment_strategy: str = Form("char", enum=["char", "sentence"]),
    max_chars: int = Form(60, gt=10, le=200)
):
    """
    Accepts an audio file and transcription parameters.
    - **segment_strategy**: 'char' for word-safe character limit, 'sentence' for sentence-based splits.
    - **max_chars**: Maximum characters per segment (used with 'char' strategy). Must be between 10 and 200.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Please upload an audio file."
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        asr_model = app_state.get("asr_model")
        result = transcribe_audio(temp_path, asr_model, segment_strategy, max_chars)
        return JSONResponse(content=result)
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in the transcribe endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()

# --- Main Execution ---

if __name__ == "__main__":
    logger.info("Starting Parakeet ASR FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)