import os
import glob
import pandas as pd
from .io import parse_trackmate_xml, extract_image_metadata

def process_trackmate_folder(folder, out_root):
    """
    Process all TrackMate XML files in a folder and export combined stats to TSV.
    """
    all_stats = []
    os.makedirs(out_root, exist_ok=True)

    for xml_path in glob.glob(os.path.join(folder, "*.xml")):
        fname = os.path.basename(xml_path)
        print(f"Processing {fname} ...")
        meta = extract_image_metadata(xml_path)
        spots_df, track_stats = parse_trackmate_xml(xml_path)
        track_stats = track_stats.copy()
        track_stats["xml_file"] = fname
        track_stats["width"] = meta.get("width")
        track_stats["height"] = meta.get("height")
        all_stats.append(track_stats)

    if not all_stats:
        print("No XML files found or no tracks extracted.")
        return pd.DataFrame()

    merged = pd.concat(all_stats, ignore_index=True)
    summary_path = os.path.join(out_root, "dataset_summary.tsv")
    merged.to_csv(summary_path, sep="\t", index=False)
    print(f"Saved merged stats to {summary_path}")
    return merged
