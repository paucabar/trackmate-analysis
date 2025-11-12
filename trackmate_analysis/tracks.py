import pandas as pd

def filter_spots_in_tracks(spots_df, track_stats):
    """Keep only spots assigned to valid tracks."""
    valid_ids = set(track_stats.index)
    return spots_df[spots_df["track_id"].isin(valid_ids)].copy()


def filter_spots_by_duration(spots_df, track_stats, min_duration=None, max_duration=None):
    """Filter out spots belonging to tracks outside a duration range."""
    mask = pd.Series(True, index=track_stats.index)
    if min_duration is not None:
        mask &= track_stats["duration_frames"] >= min_duration
    if max_duration is not None:
        mask &= track_stats["duration_frames"] <= max_duration
    valid_ids = set(track_stats[mask].index)
    return spots_df[spots_df["track_id"].isin(valid_ids)].copy()


def filter_tracks_by_duration(track_stats, min_duration=None, max_duration=None):
    """Keep only tracks within duration limits."""
    mask = pd.Series(True, index=track_stats.index)
    if min_duration is not None:
        mask &= track_stats["duration_frames"] >= min_duration
    if max_duration is not None:
        mask &= track_stats["duration_frames"] <= max_duration
    return track_stats[mask].copy()
