import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def _localname(tag):
    """Return XML local name without namespace."""
    return tag.split('}')[-1] if '}' in tag else tag


def parse_trackmate_xml(xml_path):
    """
    Parse a TrackMate XML file to extract spots and track statistics.

    Returns
    -------
    spots_df : pd.DataFrame
        Contains spot features (frame, position, intensities, etc.).
    track_stats : pd.DataFrame
        Per-track summary statistics (duration, displacement, intensity, etc.).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- collect all spots ---
    spots = []
    for elem in root.iter():
        if _localname(elem.tag) == 'Spot':
            a = elem.attrib
            sid = int(a.get('ID') or a.get('id', -1))
            frame = int(a.get('FRAME') or a.get('frame') or a.get('POSITION_T', 0))
            rec = {'spot_id': sid, 'frame': frame}
            for k, v in a.items():
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
            spots.append(rec)

    spots_df = pd.DataFrame(spots)
    if spots_df.empty:
        raise ValueError(f"No <Spot> elements found in {xml_path}")

    # --- map spot_id → track_id ---
    spot_to_track = {}
    for elem in root.iter():
        if _localname(elem.tag) == 'Track':
            track_id = int(elem.attrib.get('TRACK_ID', -1))
            for child in elem.iter():
                lname = _localname(child.tag)
                if lname == 'SpotRef':
                    sid = child.attrib.get('ID') or child.attrib.get('id')
                    if sid:
                        spot_to_track[int(sid)] = track_id
                elif lname == 'Edge':
                    s = child.attrib.get('SPOT_SOURCE_ID') or child.attrib.get('source')
                    t = child.attrib.get('SPOT_TARGET_ID') or child.attrib.get('target')
                    if s:
                        spot_to_track[int(s)] = track_id
                    if t:
                        spot_to_track[int(t)] = track_id

    spots_df["track_id"] = spots_df["spot_id"].map(spot_to_track).astype("Int64")

    # --- per-track summary stats ---
    has_track = spots_df["track_id"].notna()
    grouped = spots_df[has_track].groupby("track_id", sort=True)

    def compute_per_track(df):
        df_sorted = df.sort_values("frame")
        start_frame = df_sorted["frame"].iloc[0]
        end_frame = df_sorted["frame"].iloc[-1]
        duration = int(end_frame - start_frame + 1)
        n_spots = len(df_sorted)

        # dynamically detect intensity channels
        intensity_cols = [c for c in df_sorted.columns if "MEAN_INTENSITY" in c]
        if intensity_cols:
            mean_int = df_sorted[intensity_cols].mean(axis=1).mean()
            max_int = df_sorted[intensity_cols].max(axis=1).max()
        else:
            mean_int = np.nan
            max_int = np.nan

        coords = df_sorted[
            [c for c in ["POSITION_X", "POSITION_Y", "POSITION_Z"] if c in df_sorted.columns]
        ].to_numpy(dtype=float, copy=False)

        if coords.shape[0] >= 2:
            step_dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            path_length = float(step_dists.sum())
            mean_step = float(np.nanmean(step_dists))
            displacement = float(np.linalg.norm(coords[-1] - coords[0]))
        else:
            path_length = 0.0
            mean_step = np.nan
            displacement = 0.0

        return pd.Series({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration": duration,
            "n_spots": n_spots,
            "mean_intensity": mean_int,
            "max_intensity": max_int,
            "path_length": path_length,
            "mean_step_length": mean_step,
            "displacement": displacement,
        })

    track_stats = grouped.apply(compute_per_track).reset_index().set_index("track_id")
    return spots_df, track_stats


def extract_image_metadata(xml_path):
    """Extract <ImageData> metadata from TrackMate XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata = {}
    for elem in root.iter():
        if elem.tag.endswith("ImageData"):
            for k, v in elem.attrib.items():
                try:
                    if "." in v or "e" in v.lower():
                        metadata[k.lower()] = float(v)
                    else:
                        metadata[k.lower()] = int(v)
                except Exception:
                    metadata[k.lower()] = v
            break
    return metadata


def extract_edges_from_xml(xml_path):
    """Return DataFrame with columns ['source','target'] for spot edges."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    edges = []
    for elem in root.iter():
        if _localname(elem.tag) == "Edge":
            s = elem.attrib.get("SPOT_SOURCE_ID") or elem.attrib.get("source")
            t = elem.attrib.get("SPOT_TARGET_ID") or elem.attrib.get("target")
            if s and t:
                edges.append({"source": int(s), "target": int(t)})
    return pd.DataFrame(edges)


def extract_track_filters(xml_path):
    """
    Extract track filters.

    Parameters
    ----------
    xml_path : str
        Path to a TrackMate XML file.

    Returns
    -------
    filters_df : pd.DataFrame
        Columns: ['feature', 'value', 'is_above']
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filters = []
    for tfc in root.iter():
        if tfc.tag.endswith("TrackFilterCollection"):
            for filt in tfc.findall(".//Filter"):
                filters.append({
                    "feature": filt.attrib.get("feature"),
                    "value": float(filt.attrib.get("value", "nan")),
                    "is_above": filt.attrib.get("isabove", "true").lower() == "true",
                })

    return pd.DataFrame(filters)

