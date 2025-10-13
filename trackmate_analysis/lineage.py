import pandas as pd
import networkx as nx
from .io import _localname

def detect_split_events(spots_df, edges_df):
    """Detect spots that divide into multiple child spots."""
    if edges_df is None or edges_df.empty:
        return pd.DataFrame(columns=["split_spot", "frame", "parent_track", "child_spots", "child_tracks"])

    target_counts = edges_df.groupby("source")["target"].nunique()
    split_sources = target_counts[target_counts > 1].index.tolist()

    frame_map = spots_df.set_index("spot_id")["frame"].to_dict()
    spot2track_map = spots_df.set_index("spot_id")["track_id"].to_dict()

    rows = []
    for src in split_sources:
        child_spots = edges_df.loc[edges_df["source"] == src, "target"].unique().tolist()
        frame = frame_map.get(src)
        parent_track = spot2track_map.get(src)
        child_tracks = [spot2track_map.get(c, pd.NA) for c in child_spots]

        rows.append({
            "split_spot": int(src),
            "frame": int(frame) if frame is not None else pd.NA,
            "parent_track": int(parent_track) if not pd.isna(parent_track) else pd.NA,
            "child_spots": [int(c) for c in child_spots],
            "child_tracks": [int(ct) if not pd.isna(ct) else pd.NA for ct in child_tracks],
        })

    return pd.DataFrame(rows)


def promote_split_children(spots_df, edges_df):
    """Assign new track IDs for child branches after a split event."""
    spots_df = spots_df.copy()
    spots_df["is_pseudo"] = False
    max_tid = int(spots_df["track_id"].dropna().max()) if spots_df["track_id"].notna().any() else 0
    new_assignments = {}
    pseudo_tids = set()

    for parent, group in edges_df.groupby("source"):
        child_spots = group["target"].tolist()
        if len(child_spots) <= 1:
            continue

        child_tracks = spots_df.set_index("spot_id").loc[child_spots, "track_id"].dropna().unique()
        if len(child_tracks) == 1:
            for child in child_spots:
                max_tid += 1
                new_tid = max_tid
                pseudo_tids.add(new_tid)

                # propagate reassignment recursively
                to_reassign = {child}
                stack = [child]
                while stack:
                    cur = stack.pop()
                    for nxt in edges_df.loc[edges_df["source"] == cur, "target"]:
                        if nxt not in to_reassign:
                            to_reassign.add(nxt)
                            stack.append(nxt)
                for sid in to_reassign:
                    new_assignments[sid] = new_tid

    if new_assignments:
        spots_df.loc[spots_df["spot_id"].isin(new_assignments.keys()), "track_id"] = \
            spots_df["spot_id"].map(new_assignments).fillna(spots_df["track_id"])
        spots_df.loc[spots_df["track_id"].isin(pseudo_tids), "is_pseudo"] = True

    return spots_df

def build_lineage_graph(spots_df, xml_path):
    """
    Build lineage graph at track level from TrackMate XML.
    Adds an edge parent_track > child_track whenever a spot in one
    track connects to a spot in another (captures splits).
    """
    try:
        from lxml import etree as ET
    except Exception:
        import xml.etree.ElementTree as ET

    import networkx as nx
    G = nx.DiGraph()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ensure all tracks are in the graph
    for tid in spots_df['track_id'].dropna().unique():
        G.add_node(int(tid))

    # collect all edges: parent spot > child spot
    spot_edges = []
    for elem in root.iter():
        if _localname(elem.tag) == "Edge":
            s = elem.attrib.get("SPOT_SOURCE_ID")
            t = elem.attrib.get("SPOT_TARGET_ID")
            if s and t:
                spot_edges.append((int(s), int(t)))

    # map to track-level edges
    for s, t in spot_edges:
        ts = spots_df.loc[spots_df["spot_id"] == s, "track_id"]
        tt = spots_df.loc[spots_df["spot_id"] == t, "track_id"]
        if ts.empty or tt.empty:
            continue
        st, tt = int(ts.iloc[0]), int(tt.iloc[0])
        if st != tt:
            G.add_edge(st, tt)

    return G

def build_track_lineage_from_splits(spots_df, edges_df):
    """Construct lineage graph using split and cross-track connections."""
    G = nx.DiGraph()
    unique_tracks = [int(t) for t in pd.unique(spots_df["track_id"].dropna())]
    for t in unique_tracks:
        G.add_node(t)

    spot2track_map = spots_df.set_index("spot_id")["track_id"].to_dict()
    splits_df = detect_split_events(spots_df, edges_df)

    for _, r in splits_df.iterrows():
        parent = r["parent_track"]
        if pd.isna(parent):
            continue
        for ct in r["child_tracks"]:
            if not pd.isna(ct) and parent != ct:
                G.add_edge(int(parent), int(ct), reason="split")

    for _, e in edges_df.iterrows():
        s, t = int(e["source"]), int(e["target"])
        ps, pt = spot2track_map.get(s, pd.NA), spot2track_map.get(t, pd.NA)
        if not pd.isna(ps) and not pd.isna(pt) and ps != pt:
            G.add_edge(int(ps), int(pt), reason="edge")

    return G, splits_df
