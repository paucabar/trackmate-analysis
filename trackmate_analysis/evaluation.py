import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Custom TRA-based metric for centroid-only tracks

def match_points(gt_pts, pred_pts, distance_thresh):
    """
    Match ground truth and predicted centroids within distance_thresh using nearest neighbour.

    Parameters
    ----------
    gt_pts : ndarray (N_gt, 3 or 2)
    pred_pts : ndarray (N_pred, 3 or 2)
    distance_thresh : float

    Returns
    -------
    matches : list of (gt_idx, pred_idx)
    unmatched_gt : list of gt indices
    unmatched_pred : list of pred indices
    """
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return [], list(range(len(gt_pts))), list(range(len(pred_pts)))

    tree = cKDTree(pred_pts)
    dists, idxs = tree.query(gt_pts, distance_upper_bound=distance_thresh)

    matches, unmatched_gt, unmatched_pred = [], [], set(range(len(pred_pts)))
    for i, (d, j) in enumerate(zip(dists, idxs)):
        if np.isfinite(d) and d < distance_thresh:
            matches.append((i, j))
            unmatched_pred.discard(j)
        else:
            unmatched_gt.append(i)
    unmatched_pred = list(unmatched_pred)
    return matches, unmatched_gt, unmatched_pred


def compute_tra(gt_spots, pred_spots, gt_edges=None, pred_edges=None,
                distance_thresh=5.0, weights=None):
    """
    Compute a centroid-based approximate TRA metric between ground truth and prediction.

    Parameters
    ----------
    gt_spots, pred_spots : pd.DataFrame
        Must contain columns ['frame','POSITION_X','POSITION_Y','track_id']
        (and 'POSITION_Z' if 3D)
    gt_edges, pred_edges : pd.DataFrame
    distance_thresh : float
        Maximum centroid distance (in same units as coordinates) for a match.
    weights : dict, optional
        e.g., {'FN':1, 'FP':1, 'IDS':1, 'LNK':1}

    Returns
    -------
    results : dict with keys
        {'TRA','FN','FP','IDS','LNK','matches'}
    """

    if weights is None:
        weights = {'FN':1.0, 'FP':1.0, 'IDS':1.0, 'LNK':1.0}

    # ensure consistent columns
    coord_cols = [c for c in ['POSITION_X','POSITION_Y','POSITION_Z']
                  if c in gt_spots.columns and c in pred_spots.columns]

    gt_spots = gt_spots.copy()
    pred_spots = pred_spots.copy()

    # Storage
    total_FN, total_FP, total_IDS, total_LNK = 0, 0, 0, 0
    total_GT = len(gt_spots)
    frame_matches = {}

    # Match centroids frame by frame
    all_frames = sorted(set(gt_spots['frame'].unique()) |
                        set(pred_spots['frame'].unique()))

    for frame in all_frames:
        gt_f = gt_spots[gt_spots['frame']==frame].reset_index(drop=True)
        pred_f = pred_spots[pred_spots['frame']==frame].reset_index(drop=True)
        matches, um_gt, um_pr = match_points(
            gt_f[coord_cols].to_numpy(),
            pred_f[coord_cols].to_numpy(),
            distance_thresh
        )
        pairs = [(int(gt_f.loc[i,'track_id']), int(pred_f.loc[j,'track_id']))
                for i,j in matches]
        frame_matches[frame] = pairs
        total_FN += len(um_gt)
        total_FP += len(um_pr)


    # Detect identity switches
    gt_to_pred = {}
    for f in sorted(frame_matches.keys()):
        for gt_tid, pr_tid in frame_matches[f]:
            prev = gt_to_pred.get(gt_tid)
            if prev is None:
                gt_to_pred[gt_tid] = pr_tid
            elif prev != pr_tid:
                total_IDS += 1
                gt_to_pred[gt_tid] = pr_tid

    # Compare lineage links (parent–child relationships) on matched spots
    if gt_edges is not None and pred_edges is not None:
        # Build mapping: frame, gt_spot_idx -> pred_spot_idx
        spot_matches = {}
        for f, pairs in frame_matches.items():
            gt_f = gt_spots[gt_spots['frame'] == f]
            pr_f = pred_spots[pred_spots['frame'] == f]
            # build reverse lookup by track_id
            gt_by_tid = gt_f.set_index('track_id')['spot_id'].to_dict()
            pr_by_tid = pr_f.set_index('track_id')['spot_id'].to_dict()
            for gt_tid, pr_tid in pairs:
                if gt_tid in gt_by_tid and pr_tid in pr_by_tid:
                    spot_matches[gt_by_tid[gt_tid]] = pr_by_tid[pr_tid]

        gt_edge_set = set(tuple(x) for x in gt_edges[['source','target']].itertuples(index=False))
        pr_edge_set = set(tuple(x) for x in pred_edges[['source','target']].itertuples(index=False))

        matched_edges = 0
        missed_edges = 0
        extra_edges = 0

        # Check for GT edges that are not respected in prediction
        for s_gt, t_gt in gt_edge_set:
            s_pr = spot_matches.get(s_gt)
            t_pr = spot_matches.get(t_gt)
            if s_pr is not None and t_pr is not None:
                if (s_pr, t_pr) in pr_edge_set:
                    matched_edges += 1
                else:
                    missed_edges += 1
            else:
                missed_edges += 1

        # Optionally: check for predicted edges that connect unmatched or wrong cells
        #for s_pr, t_pr in pr_edge_set:
        #    inv_map = {v:k for k,v in spot_matches.items()}
        #    s_gt = inv_map.get(s_pr)
        #    t_gt = inv_map.get(t_pr)
        #    if s_gt is not None and t_gt is not None:
        #        if (s_gt, t_gt) not in gt_edge_set:
        #            extra_edges += 1
        #    else:
        #        extra_edges += 1

        total_LNK = missed_edges# + extra_edges


    # Compute weighted error sum
    num = (weights['FN']*total_FN +
           weights['FP']*total_FP +
           weights['IDS']*total_IDS +
           weights['LNK']*total_LNK
           )

    # Compute TRA metric
    N = len(gt_spots)
    M = len(gt_edges) if gt_edges is not None else 0

    TRA_nodes = max(0.0, 1.0 - num / max(1, N))
    TRA_nodes_edges = max(0.0, 1.0 - num / max(1, N + M))

    print(f"FN={total_FN}, FP={total_FP}, IDS={total_IDS}, LNK={total_LNK}")
    print(f"TRA (nodes) = {TRA_nodes:.4f}")
    print(f"TRA (nodes+edges) = {TRA_nodes_edges:.4f}")

    return dict(TRA=TRA_nodes, FN=total_FN, FP=total_FP, IDS=total_IDS,
                LNK=total_LNK, total_GT=total_GT, matches=frame_matches)
