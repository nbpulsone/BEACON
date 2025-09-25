import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

def aggregate_tt_dr_results(
    files: List[str],
    output_path: str,
    method_name: str,
    inline_f1_std: bool = False,
    std_format: str = " ± {std:.4f}"
) -> Tuple[
    Dict[int, Dict[str, Dict[str, float]]],
    Dict[int, Dict[str, Dict[str, float]]]
]:
    """
    Aggregate multiple tt/dr result files into mean (and std) per (budget, category, metric).

    Parameters
    ----------
    files : List[str]
        Paths to input result files (each with multiple budgets, categories, and F1/Precision/Recall lines).
    output_path : str
        Where to write the aggregated file (same format as inputs; means only unless inline_f1_std=True).
    inline_f1_std : bool, default False
        If True, prints F1 as "F1: <mean> ± <std>" in the main output instead of listing stds at the end.
    std_format : str, default " ± {std:.4f}"
        String format for inlined std (only used if inline_f1_std=True).

    Returns
    -------
    aggregated : dict
        aggregated[budget][category][metric] -> mean value
    stds : dict
        stds[budget][category][metric] -> std value

    Notes
    -----
    - Expected lines:
        STARTING BUDGET: <int>
        wdc_<category> F1: <float>
        wdc_<category> Precision: <float>
        wdc_<category> Recall: <float>
    - Budgets can appear multiple times across files; values are aggregated across all files.
    """
    # Regex for category metric lines
    line_pattern = re.compile(r"^(wdc_[\w_]+)\s+(F1|Precision|Recall):\s+([+-]?\d+(?:\.\d+)?)")
    #line_pattern = re.compile(r"^updated_data/([\w_]+_test\.txt)(F1|Precision|Recall):\s+([+-]?\d+(?:\.\d+)?)")

    # Nested: results[budget][category][metric] = list of values
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    current_budget: Optional[int] = None

    # Parse all files
    for path in files:
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("STARTING BUDGET:"):
                    # e.g., "STARTING BUDGET: 1000"
                    try:
                        current_budget = int(line.split(":")[1].strip())
                    except Exception:
                        current_budget = None
                    continue

                m = line_pattern.match(line)
                if m and current_budget is not None:
                    category, metric, val = m.groups()
                    results[current_budget][category][metric].append(float(val))

    # Compute mean and std
    aggregated = defaultdict(lambda: defaultdict(dict))
    stds = defaultdict(lambda: defaultdict(dict))
    for budget, cats in results.items():
        for category, metrics in cats.items():
            for metric, values in metrics.items():
                arr = np.array(values, dtype=float)
                if arr.size == 0:
                    continue
                aggregated[budget][category][metric] = float(arr.mean())
                stds[budget][category][metric] = float(arr.std(ddof=0))  # population std consistent across seeds

    # Write aggregated output (same format as inputs)
    with open(output_path, "w") as out:
        for budget in sorted(aggregated.keys()):
            out.write("=" * 20 + f"\nSTARTING BUDGET: {budget}\n" + "=" * 20 + "\n")
            out.write(f"~~~ {method_name} {budget} ~~~\n")
            for category in sorted(aggregated[budget].keys()):
                f1_mean = aggregated[budget][category].get("F1", float("nan"))
                prec_mean = aggregated[budget][category].get("Precision", float("nan"))
                rec_mean = aggregated[budget][category].get("Recall", float("nan"))

                if inline_f1_std:
                    f1_std = stds[budget][category].get("F1", 0.0)
                    out.write(f"{category} F1: {f1_mean:.4f}{std_format.format(std=f1_std)}\n")
                else:
                    out.write(f"{category} F1: {f1_mean:.4f}\n")

                out.write(f"{category} Precision: {prec_mean:.4f}\n")
                out.write(f"{category} Recall: {rec_mean:.4f}\n")
            out.write("\n")

        """if not inline_f1_std:
            out.write("\n\nStandard Deviations of F1s:\n")
            for budget in sorted(stds.keys()):
                out.write(f"\nBUDGET {budget}:\n")
                for category in sorted(stds[budget].keys()):
                    f1_std = stds[budget][category].get("F1", 0.0)
                    out.write(f"{category} F1 std: {f1_std:.4f}\n")"""

    return aggregated, stds


if __name__ == '__main__':
    method = sys.argv[1]
    infiles = [sys.argv[i] for i in range(2, len(sys.argv) - 1)]
    outfile = sys.argv[len(sys.argv)-1]
    print(f"INFILES: {infiles}")
    print(f"OUTFILE: {outfile}")
    aggregate_tt_dr_results(infiles, outfile, method, inline_f1_std=False)