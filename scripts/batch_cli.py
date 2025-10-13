"""
Command-line interface for batch processing TrackMate XML files.

Uses the `process_trackmate_folder()` function defined in `batch.py`.

Example usage:
    python batch_cli.py --input ./data/xmls --output ./results
"""

import argparse
from trackmate_analysis.batch import process_trackmate_folder


def main():
    parser = argparse.ArgumentParser(
        description="Batch process TrackMate XML files and export combined stats to TSV."
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input folder containing TrackMate XML files."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output folder for the merged TSV summary."
    )

    args = parser.parse_args()

    # Run the existing batch processor
    process_trackmate_folder(folder=args.input, out_root=args.output)


if __name__ == "__main__":
    main()
