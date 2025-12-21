import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

GOAL_X = PITCH_LENGTH
GOAL_Y = PITCH_WIDTH / 2.0

GOAL_HALF_WIDTH = 4.0
LEFT_POST_Y = GOAL_Y - GOAL_HALF_WIDTH
RIGHT_POST_Y = GOAL_Y + GOAL_HALF_WIDTH

FEATURE_COLUMNS = [
    "distance",
    "angle",
    "minute",
    "second",
    "under_pressure",
    "first_time",
    "play_pattern",
    "body_part",
    "shot_type",
]

TARGET_COLUMN = "is_goal"


def iter_event_files(events_root: Path) -> Iterable[Path]:
    """
    Iterate over all event files (.json) under the given root, recursively.
    """
    for path in sorted(events_root.rglob("*.json")):
        yield path


def compute_distance_and_angle(location: Optional[List[float]]) -> Tuple[float, float]:
    """
    Given a location [x, y] (StatsBomb coordinates in normalized meters 0-120 x 0-80),
    compute:
    - distance to the center of the goal
    - shot angle (between the vectors to the left and right posts)
    """
    if not location or len(location) < 2:
        return float("nan"), float("nan")

    x, y = float(location[0]), float(location[1])

    dx = GOAL_X - x
    dy = GOAL_Y - y
    distance = math.hypot(dx, dy)

    left_vec = np.array([GOAL_X - x, LEFT_POST_Y - y], dtype=float)
    right_vec = np.array([GOAL_X - x, RIGHT_POST_Y - y], dtype=float)

    left_norm = np.linalg.norm(left_vec)
    right_norm = np.linalg.norm(right_vec)
    denom = left_norm * right_norm

    if denom == 0.0:
        angle = 0.0
    else:
        cos_angle = float(np.dot(left_vec, right_vec) / denom)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        angle = float(math.acos(cos_angle))

    return distance, angle


def shot_event_to_features(event: Dict) -> Dict:
    """
    Convert a StatsBomb shot event into a feature dictionary.
    """
    location = event.get("location")
    distance, angle = compute_distance_and_angle(location)

    minute = event.get("minute")
    second = event.get("second")
    under_pressure = 1 if event.get("under_pressure") else 0

    shot = event.get("shot") or {}
    first_time = 1 if shot.get("first_time") else 0

    play_pattern = None
    pp = event.get("play_pattern")
    if isinstance(pp, dict):
        play_pattern = pp.get("name")
    elif isinstance(pp, str):
        play_pattern = pp

    body_part = None
    bp = shot.get("body_part")
    if isinstance(bp, dict):
        body_part = bp.get("name")
    elif isinstance(bp, str):
        body_part = bp

    shot_type = None
    st = shot.get("type")
    if isinstance(st, dict):
        shot_type = st.get("name")
    elif isinstance(st, str):
        shot_type = st

    features = {
        "distance": distance,
        "angle": angle,
        "minute": minute,
        "second": second,
        "under_pressure": under_pressure,
        "first_time": first_time,
        "play_pattern": play_pattern,
        "body_part": body_part,
        "shot_type": shot_type,
    }

    return features


def extract_shots_from_file(path: Path) -> List[Dict]:
    """
    Read an events file and return a list of records (features + target)
    for events of type Shot only.
    """
    with path.open("r", encoding="utf-8") as f_in:
        events = json.load(f_in)

    rows: List[Dict] = []

    for event in events:
        event_type = (event.get("type") or {}).get("name")
        if event_type != "Shot":
            continue

        shot = event.get("shot") or {}
        outcome = shot.get("outcome") or {}
        outcome_name = outcome.get("name")
        is_goal = 1 if outcome_name == "Goal" else 0

        features = shot_event_to_features(event)
        row = {
            **features,
            TARGET_COLUMN: is_goal,
        }
        rows.append(row)

    return rows


def load_shots_dataframe(
    events_root: Path,
    limit_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a DataFrame with all shots found under the events folder.
    """
    all_rows: List[Dict] = []

    for i, path in enumerate(iter_event_files(events_root)):
        if limit_files is not None and i >= limit_files:
            break
        rows = extract_shots_from_file(path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    df = df.dropna(subset=["distance", "angle", "minute"])
    df = df.reset_index(drop=True)

    return df
