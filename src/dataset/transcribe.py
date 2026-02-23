"""Transcribe video audio using local Whisper or OpenAI Whisper API."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)


def transcribe_video_local(video_path: str, model_name: str = "base") -> List[TranscriptSegment]:
    """Transcribe video audio using local openai-whisper model.

    No API key needed. Downloads model on first run (~140MB for 'base').
    Supports word-level timestamps.
    """
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(video_path, word_timestamps=True)

    segments: List[TranscriptSegment] = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append(WordTimestamp(
                word=w["word"].strip(),
                start=round(w["start"], 3),
                end=round(w["end"], 3),
            ))
        segments.append(TranscriptSegment(
            start=round(seg["start"], 3),
            end=round(seg["end"], 3),
            text=seg["text"].strip(),
            words=words,
        ))
    return segments


def transcribe_video_api(video_path: str) -> List[TranscriptSegment]:
    """Transcribe video audio via OpenAI Whisper API.

    Requires OPENAI_API_KEY environment variable.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    with open(video_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    segments: List[TranscriptSegment] = []
    # API returns words at segment level or top level depending on version
    top_words = getattr(response, "words", None) or []
    for seg in (response.segments or []):
        seg_words = getattr(seg, "words", None) or []
        words = []
        for w in seg_words:
            words.append(WordTimestamp(
                word=w.word.strip(),
                start=round(w.start, 3),
                end=round(w.end, 3),
            ))
        # If segment had no words, find matching words from top-level list
        if not words and top_words:
            for w in top_words:
                if w.start >= seg.start and w.end <= seg.end:
                    words.append(WordTimestamp(
                        word=w.word.strip(),
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                    ))
        segments.append(TranscriptSegment(
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text=seg.text.strip(),
            words=words,
        ))

    return segments


def transcribe_video(video_path: str, use_api: bool = False, model_name: str = "base") -> List[TranscriptSegment]:
    """Transcribe video audio. Uses local Whisper by default, API if use_api=True."""
    if use_api:
        return transcribe_video_api(video_path)
    return transcribe_video_local(video_path, model_name=model_name)


def segments_to_dicts(segments: List[TranscriptSegment]) -> list:
    """Serialize TranscriptSegment list to JSON-serializable dicts."""
    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words],
        })
    return result
