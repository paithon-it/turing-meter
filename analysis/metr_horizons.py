"""METR's published horizons, as they are: p50 and p80 per model, with the doubling time of the suite.

Usage:
    curl -sLO https://metr.org/assets/benchmark_results_1_1.yaml
    python metr_horizons.py benchmark_results_1_0.yaml benchmark_results_1_1.yaml
"""
import sys

import yaml


def main(paths):
    for path in paths:
        d = yaml.safe_load(open(path))
        print("==", path.split("/")[-1], d.get("benchmark_name"))
        print("  doubling time (days):", d.get("doubling_time_in_days"))
        for name, v in d["results"].items():
            p50, p80 = v["metrics"]["p50_horizon_length"], v["metrics"]["p80_horizon_length"]
            print(f"  {name:45s} {str(v.get('release_date')):10s} p50 {p50['estimate']:9.2f} min ({p50['estimate']/60:6.2f} h) "
                  f"[{p50.get('ci_low', 0):.0f}-{p50.get('ci_high', 0):.0f}] | p80 {p80['estimate']:8.2f} min ({p80['estimate']/60:5.2f} h)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
