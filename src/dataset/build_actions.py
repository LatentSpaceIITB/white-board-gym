"""State machine: converts sorted ExcalidrawElements → action sequence.

Action types (matching the paper's 8 MDP primitives):
  STROKE, TEXT, ERASE, SETCOLOR, SETWIDTH, WAIT, AUDIOSYNC, FINISH
  (ADDIMAGE not yet implemented — no image elements in current demos)
"""

from typing import List, Optional
from .parse_excalidraw import ExcalidrawElement
from .normalize import Normalizer

GAP_THRESHOLD = 0.1  # seconds


def _find_active_segment(transcript: list, t: float) -> int:
    """Return the index of the transcript segment active at video time t.

    Returns the last segment whose start <= t, preferring one where
    t falls within [start, end].  Returns -1 if no segment starts before t.
    """
    best = -1
    for i, seg in enumerate(transcript):
        if seg["start"] <= t:
            best = i
            if t <= seg["end"]:
                return i
    return best


def build_actions(
    elements: List[ExcalidrawElement],
    normalizer: Normalizer,
    drawing_start_offset: float,
    video_duration: float,
    transcript: Optional[list] = None,
) -> List[dict]:
    """Convert sorted elements to a flat action sequence.

    State machine tracks current_color and current_width.

    For each element (in order):
      1. Gap check: if gap > GAP_THRESHOLD → emit WAIT(gap, prev_video_t)
      2. audioSync: if active transcript segment changed → emit AUDIOSYNC
      3. For deleted elements → emit ERASE (skip color/width changes)
      4. For live elements:
         a. Color change → emit SETCOLOR(color, video_t)
         b. Width change → emit SETWIDTH(width, video_t)
         c. Primary action: TEXT or STROKE at video_t

    After all elements:
      - Tail WAIT if video_duration - last_video_t > GAP_THRESHOLD
      - FINISH at video_duration
    """
    actions: List[dict] = []
    current_color: Optional[str] = None
    current_width: Optional[float] = None
    current_segment_idx: int = -1
    prev_video_t: float = drawing_start_offset
    prev_rel_time: float = 0.0

    for i, elem in enumerate(elements):
        video_t = drawing_start_offset + elem.rel_time

        # 1. Gap check (not for first element)
        if i > 0:
            gap = elem.rel_time - prev_rel_time
            if gap > GAP_THRESHOLD:
                actions.append({
                    "type": "WAIT",
                    "duration": round(gap, 3),
                    "timestamp": round(prev_video_t, 3),
                })

        # 2. audioSync — emit when active transcript segment changes
        if transcript:
            seg_idx = _find_active_segment(transcript, video_t)
            if seg_idx >= 0 and seg_idx != current_segment_idx:
                actions.append({
                    "type": "AUDIOSYNC",
                    "segment_index": seg_idx,
                    "text": transcript[seg_idx]["text"],
                    "timestamp": round(video_t, 3),
                })
                current_segment_idx = seg_idx

        # 3. Deleted element → ERASE
        if elem.is_deleted:
            cx = elem.x + elem.width / 2
            cy = elem.y + elem.height / 2
            norm_w = max(1, normalizer.norm_x(elem.x + elem.width) - normalizer.norm_x(elem.x))
            norm_h = max(1, normalizer.norm_y(elem.y + elem.height) - normalizer.norm_y(elem.y))
            actions.append({
                "type": "ERASE",
                "x": normalizer.norm_x(cx),
                "y": normalizer.norm_y(cy),
                "width": norm_w,
                "height": norm_h,
                "timestamp": round(video_t, 3),
            })
        else:
            # 4a. Color change
            if elem.stroke_color != current_color:
                actions.append({
                    "type": "SETCOLOR",
                    "color": elem.stroke_color,
                    "timestamp": round(video_t, 3),
                })
                current_color = elem.stroke_color

            # 4b. Width change
            if elem.stroke_width != current_width:
                actions.append({
                    "type": "SETWIDTH",
                    "width": elem.stroke_width,
                    "timestamp": round(video_t, 3),
                })
                current_width = elem.stroke_width

            # 4c. Primary action
            if elem.type == "text":
                cx = elem.x + elem.width / 2
                cy = elem.y + elem.height / 2
                font_size_norm = max(6, int((elem.font_size or 20) / normalizer.dy * 255))
                actions.append({
                    "type": "TEXT",
                    "content": elem.text or "",
                    "x": normalizer.norm_x(cx),
                    "y": normalizer.norm_y(cy),
                    "font_size": elem.font_size,
                    "font_size_norm": font_size_norm,
                    "timestamp": round(video_t, 3),
                })
            elif elem.type == "freedraw":
                pts = [normalizer.norm_point(elem.x + p[0], elem.y + p[1]) for p in elem.points]
                actions.append({
                    "type": "STROKE",
                    "points": pts,
                    "color": elem.stroke_color,
                    "width": elem.stroke_width,
                    "timestamp": round(video_t, 3),
                })

        prev_video_t = video_t
        prev_rel_time = elem.rel_time

    # Tail gap
    tail_gap = video_duration - prev_video_t
    if tail_gap > GAP_THRESHOLD:
        actions.append({
            "type": "WAIT",
            "duration": round(tail_gap, 3),
            "timestamp": round(prev_video_t, 3),
        })

    # Finish
    actions.append({
        "type": "FINISH",
        "timestamp": round(video_duration, 3),
    })

    return actions
