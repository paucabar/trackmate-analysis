import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def plot_trajectories_old(spots_df, width=None, height=None, ax=None, figsize=(6,6), show_labels=False):
    """Plot XY trajectories, optionally fit to original image dimensions."""
    if {"POSITION_X", "POSITION_Y"}.issubset(spots_df.columns):
        xcol, ycol = "POSITION_X", "POSITION_Y"
    elif {"x", "y"}.issubset(spots_df.columns):
        xcol, ycol = "x", "y"
    else:
        raise KeyError("No position columns found in DataFrame.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    untracked = spots_df[spots_df["track_id"].isna()]
    if not untracked.empty:
        ax.scatter(untracked[xcol], untracked[ycol], s=6, alpha=0.6, label="untracked")

    tracks = spots_df.dropna(subset=["track_id"]).groupby("track_id", sort=True)
    cmap = plt.get_cmap("tab20")
    for i, (tid, df) in enumerate(tracks):
        df = df.sort_values("frame")
        color = cmap(i % cmap.N)
        ax.plot(df[xcol], df[ycol], marker="o", markersize=3, linewidth=1, alpha=0.9, color=color)
        if show_labels:
            ax.text(df[xcol].iloc[0], df[ycol].iloc[0], str(tid), fontsize=7, color=color)

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Trajectories (XY)")
    ax.set_aspect("equal", adjustable="box")

    if width is not None and height is not None:
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

    plt.tight_layout()
    return fig, ax

def plot_trajectories(spots_df, edges_df=None, width=None, height=None,
                      ax=None, figsize=(6,6), show_labels=False):
    """
    Plot XY trajectories with optional lineage edges.

    If `edges_df` is provided, only edges are drawn between linked spots
    (no implicit line between sequential points in a track).
    Otherwise, tracks are connected by frame order.
    """

    # Determine coordinate columns
    if {"POSITION_X", "POSITION_Y"}.issubset(spots_df.columns):
        xcol, ycol = "POSITION_X", "POSITION_Y"
    elif {"x", "y"}.issubset(spots_df.columns):
        xcol, ycol = "x", "y"
    else:
        raise KeyError("No position columns found in DataFrame.")

    # Prepare axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Untracked spots
    untracked = spots_df[spots_df["track_id"].isna()]
    if not untracked.empty:
        ax.scatter(untracked[xcol], untracked[ycol], s=6, alpha=0.6, label="untracked")

    # Prepare color mapping
    tracks = spots_df.dropna(subset=["track_id"]).groupby("track_id", sort=True)
    cmap = plt.get_cmap("tab20")
    track_colors = {tid: cmap(i % cmap.N) for i, (tid, _) in enumerate(tracks)}

    # Plot spots only
    for tid, df in tracks:
        color = track_colors[tid]
        ax.scatter(df[xcol], df[ycol], s=10, color=color, alpha=0.9)
        if show_labels:
            ax.text(df[xcol].iloc[0], df[ycol].iloc[0], str(tid),
                    fontsize=7, color=color)

    # Plot edges (if provided)
    if edges_df is not None and not edges_df.empty:
        spot_lookup = spots_df.set_index("spot_id")[[xcol, ycol, "track_id"]]
        for _, row in edges_df.iterrows():
            s, t = row["source"], row["target"]
            if s in spot_lookup.index and t in spot_lookup.index:
                xs, ys = spot_lookup.loc[s, [xcol, ycol]]
                xt, yt = spot_lookup.loc[t, [xcol, ycol]]
                tid = spot_lookup.loc[s, "track_id"]
                color = track_colors.get(tid, "gray")
                ax.plot([xs, xt], [ys, yt], color=color,
                        linewidth=1, alpha=0.6, zorder=0)

    # Final formatting
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Trajectories (XY)")
    ax.set_aspect("equal", adjustable="box")

    if width is not None and height is not None:
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

    plt.tight_layout()
    return fig, ax


def plot_track_histograms(track_stats):
    """Plot histograms for track duration and path length."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(track_stats["duration"].dropna(), bins=20)
    axs[0].set_title("Track duration (frames)")
    axs[0].set_xlabel("frames")
    axs[1].hist(track_stats["path_length"].dropna(), bins=20)
    axs[1].set_title("Path length (units)")
    axs[1].set_xlabel("units")
    plt.tight_layout()
    return fig, axs


def plot_features_over_time(
    spots_df,
    track_id,
    features,
    lineage_graph=None,
    aggregate="mean",
    nframes=None,
):
    """ 
    Plot feature evolution over time for a given track and its descendants.
    If nframes (from metadata) is provided, the x-axis spans [0, nframes - 1].

    Parameters
    ----------
    spots_df : pd.DataFrame
        Must contain ['track_id', 'frame'] and the feature columns.
    track_id : int
        Track to plot (plus its descendants if lineage_graph is given).
    features : str or list of str
        Feature(s) to plot over time.
    lineage_graph : networkx.DiGraph, optional
        If given, descendants of track_id will also be plotted.
    aggregate : {'mean', 'median'}, default 'mean'
        Aggregation per frame.
    nframes : int, optional
        Total number of frames in the experiment (from metadata["nframes"]).
        If None, will use min–max of available frames.
    """

    if isinstance(features, str):
        features = [features]

    # Get full set of tracks (parent + descendants)
    tracks = [track_id]
    if lineage_graph is not None and track_id in lineage_graph.nodes:
        tracks.extend(sorted(nx.descendants(lineage_graph, track_id)))

    # Time axis: use metadata if available
    if nframes is not None and nframes > 0:
        time_index = pd.Index(range(nframes), name="frame")
    else:
        tmin, tmax = spots_df["frame"].min(), spots_df["frame"].max()
        time_index = pd.Index(range(tmin, tmax + 1), name="frame")

    # Create figure
    fig, axs = plt.subplots(len(features), 1, sharex=True, figsize=(8, 3 * len(features)))
    if len(features) == 1:
        axs = [axs]

    cmap = plt.get_cmap("tab10")

    for ax, feat in zip(axs, features):
        if feat not in spots_df.columns:
            ax.text(
                0.5, 0.5,
                f"Feature '{feat}' not found",
                ha="center", va="center",
                transform=ax.transAxes
            )
            continue

        for i, tid in enumerate(tracks):
            grp = spots_df[spots_df["track_id"] == tid]
            if grp.empty:
                continue

            if aggregate == "mean":
                series = grp.groupby("frame")[feat].mean().reindex(time_index)
            else:
                series = grp.groupby("frame")[feat].median().reindex(time_index)

            ax.plot(
                series.index, series.values,
                marker="o", label=f"Track {tid}",
                color=cmap(i % cmap.N)
            )

        ax.set_ylabel(feat)
        ax.legend(fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.3)

    axs[-1].set_xlabel("Frame")
    axs[-1].set_xlim(0, (nframes - 1) if nframes else time_index.max())
    plt.tight_layout()
    return fig, axs

