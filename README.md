# TrackMate Analysis

A lightweight Python toolkit to **parse and analyze TrackMate XML files** from cell tracking experiments.  
It provides tools to extract image metadata, track and spot statistics, and basic visualization utilities.

---

## Installation

Create and activate a conda environment:

```bash
conda create -n trackmate_env python=3.10 -y
conda activate trackmate_env
```

Then install dependencies:

```bash
pip install pandas matplotlib networkx lxml
```

Clone or copy this repository:

```bash
git clone https://github.com/paucabar/trackmate-analysis
cd trackmate-analysis
```

## Example Usage

- See [`notebooks/example_notebook.ipynb`](notebooks/example_notebook.ipynb) for a step-by-step demonstration of how to parse and plot track data from a single TrackMate XML file.

- See [`scripts/batch_cli.py`](scripts/batch_cli.py) for command-line batch processing to generate merged track statistics (`dataset_summary.tsv`) for all XML files in a folder.

---

## Citation

This package is a lightweight post-processing tool that extracts and analyzes data from TrackMate output files.  
If your data were produced using **TrackMate**, please cite the following publications:

> **Tinevez, J.Y., Perry, N., Schindelin, J., et al. (2017).**  
> *TrackMate: An open and extensible platform for single-particle tracking.*  
> *Methods* 115: 80–90.  
> [https://doi.org/10.1016/j.ymeth.2016.09.016](https://doi.org/10.1016/j.ymeth.2016.09.016)

> **Ershov, D., Phan, M.S., Pylvänäinen, J.W., et al. (2022).**  
> *TrackMate 7: integrating state-of-the-art segmentation algorithms into tracking pipelines.*  
> *Nature Methods* 19: 829–832.  
> [https://doi.org/10.1038/s41592-022-01507-1](https://doi.org/10.1038/s41592-022-01507-1)
