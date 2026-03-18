import argparse
import pandas as pd
from skimage import io
from pathlib import Path
from tqdm import tqdm


def check_tiles(csv_path, expected_size=512, cols=("x_location", "y_location")):
    """
    Traverse dataset CSV and check whether all PNG tiles
    in specified columns have shape (expected_size, expected_size).

    Returns:
        bad_files: list of dicts with mismatch info
    """
    df = pd.read_csv(csv_path)
    bad_files = []

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV")

        print(f"\nChecking column: {col}")

        for path in tqdm(df[col].dropna().unique()):
            p = Path(path)

            if not p.exists():
                bad_files.append({
                    "column": col,
                    "path": str(p),
                    "error": "FILE_NOT_FOUND"
                })
                continue

            try:
                img = io.imread(p)

                # handle grayscale / RGB / RGBA
                if img.ndim == 2:
                    h, w = img.shape
                else:
                    h, w = img.shape[:2]

                if (h, w) != (expected_size, expected_size):
                    bad_files.append({
                        "column": col,
                        "path": str(p),
                        "shape": (h, w)
                    })

            except Exception as e:
                bad_files.append({
                    "column": col,
                    "path": str(p),
                    "error": repr(e)
                })

    return bad_files


def main():
    parser = argparse.ArgumentParser(
        description="Check whether tiles in dataset CSV have expected size"
    )
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--expected_size", type=int, default=512,
                        help="Expected tile size (default: 512)")
    args = parser.parse_args()

    bad_files = check_tiles(
        csv_path=args.csv,
        expected_size=args.expected_size
    )

    print("\n================ SUMMARY ================")
    if not bad_files:
        print(" All tiles match expected size.")
    else:
        print(f" Found {len(bad_files)} problematic tiles:\n")
        for item in bad_files[:20]:
            print(item)
        if len(bad_files) > 20:
            print(f"... ({len(bad_files)-20} more)")

        # optional: save report
        out = Path("tile_size_mismatch_report.csv")
        pd.DataFrame(bad_files).to_csv(out, index=False)
        print(f"\n Full report saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
