"""Launch the five manuscript seeds for one privacy budget."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 123, 456, 789, 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()
    trainer = Path(__file__).resolve().parents[1] / "dp_pipeline" / "train.py"
    for seed in SEEDS:
        destination = Path(args.output) / f"seed_{seed}"
        subprocess.run(
            [
                sys.executable,
                str(trainer),
                "--data", args.data,
                "--output", str(destination),
                "--num-classes", str(args.num_classes),
                "--epsilon", str(args.epsilon),
                "--delta", str(args.delta),
                "--public-seed", str(seed),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
