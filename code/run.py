import argparse
import subprocess
import csv
import re
from itertools import product

parser = argparse.ArgumentParser(
    prog="Program Executor",
    description="This program is in charge of executing a CUDA program with multiple parameters and storing their results",
)


parser.add_argument("-r", "--results", required=True, help="Path to the results file")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

args = parser.parse_args()

def _print(message: str) -> None:
    if args.verbose:
        print(message)

def extract_time(output: str) -> float:
    match = re.search(r"Execution Time\s*=\s*([\d.]+)ms", output)

    if match:
        return float(match.group(1))
    return -1


def main() -> None:

    block_dim = [
        (4, 256),
        (8, 126),
        (16, 64),
        (32, 32),
        (64, 16),
        (128, 8),
        (256, 4)
    ]
    steps_dim = [100, 1_000, 10_000, 100_000]
    size_dim = [100, 1_000, 2_000]
    
    combinations = list(product(block_dim, steps_dim, size_dim))

    with open(args.results, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Block Dim", "Size", "Steps", "Time"])

        for (block_x, block_y), steps, size in combinations:
            output_filename = f"./results/out_b{block_x}x{block_y}_s{size}_t{steps}.bmp"

            command = [
                "./heat_cuda",
                str(block_x), str(block_y), str(size), str(steps), output_filename
            ]
            
            _print(f"Running: {" ".join(command)}")
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                _print(result.stdout)
                time = extract_time(result.stdout)
                _print(f"{time}\n")

                writer.writerow([block_x, block_y, size, steps, time])

            except subprocess.CalledProcessError as e:
                print(f"Error during execution: {e}")
                exit(1)


if __name__ == "__main__":
    main()