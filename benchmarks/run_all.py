import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from benchmarks.utils import (
    get_env_info,
    RESULTS_DIR,
    FROZEN_DEV_THRESHOLD,
)

from benchmarks.t1_eval_generalization import run_t1
from benchmarks.t2_per_attack import run_t2
from benchmarks.t3_full_metrics import run_t3
from benchmarks.t4_baselines import run_t4
from benchmarks.t5_seed_variance import run_t5
from benchmarks.t6_robustness import run_t6
from benchmarks.t7_ood_generators import run_t7
from benchmarks.t8_non_speech_sanity import run_t8
from benchmarks.t9_calibration import run_t9
from benchmarks.t10_threshold_sensitivity import run_t10
from benchmarks.t11_api_latency import run_t11
from benchmarks.t12_failure_analysis import run_t12

def main():
    parser = argparse.ArgumentParser(description="Voice Deepfake Detector Comprehensive Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run rapid evaluation with compact subsampling")
    parser.add_argument("--full", action="store_true", help="Run full 71k evaluation set (Warning: takes ~10 hours on CPU)")
    args = parser.parse_args()

    mode = "quick" if args.quick else ("full" if args.full else "standard (stratified 3000)")
    print("=" * 70)
    print(" VOICE DEEPFAKE DETECTOR BENCHMARK SUITE")
    print(f" Mode: {mode.upper()}")
    print("=" * 70)

    env_info = get_env_info()
    print("Environment Details:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")
    print(f"Decision Threshold Applied: {FROZEN_DEV_THRESHOLD} (on spoof probability)")
    print("=" * 70)

    start_total = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. T1: Eval Generalization
    t1_frozen, t1_05 = run_t1(quick=args.quick, full=args.full)

    # 2. T2: Per-Attack Breakdown
    t2_df = run_t2(quick=args.quick, full=args.full)

    # 3. T3: Full Metrics & Curves
    t3_df = run_t3(quick=args.quick, full=args.full)

    # 4. T4: Baselines Comparison
    t4_df = run_t4(quick=args.quick, full=args.full)

    # 5. T5: Seed Variance (5 seeds)
    t5_df = run_t5(quick=args.quick, full=args.full)

    # 6. T6: Robustness & Audio Perturbations
    t6_df = run_t6(quick=args.quick, full=args.full)

    # 7. T7: Out-of-Domain Modern Voice Generators
    t7_res = run_t7(quick=args.quick, full=args.full)

    # 8. T8: Non-Speech Sanity Tests
    t8_df = run_t8(quick=args.quick, full=args.full)

    # 9. T9: Calibration Analysis
    t9_df = run_t9(quick=args.quick, full=args.full)

    # 10. T10: Threshold Sensitivity & Shift
    t10_res = run_t10(quick=args.quick, full=args.full)

    # 11. T11: API Latency & Concurrency
    t11_res = run_t11(quick=args.quick, full=args.full)

    # 12. T12: Failure Analysis
    t12_df = run_t12(quick=args.quick, full=args.full)

    total_duration = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"ALL BENCHMARKS COMPLETE in {total_duration:.1f} s ({total_duration/60.0:.2f} min)")
    print("=" * 70)

    # Build machine-readable summary.json
    summary = {
        "benchmark_metadata": {
            "mode": mode,
            "duration_seconds": round(total_duration, 2),
            "environment": env_info,
            "frozen_dev_threshold": FROZEN_DEV_THRESHOLD,
        },
        "dev_vs_eval_generalization": {
            "dev_eer_percent": round(float(t10_res["dev_optimal_eer"] * 100), 3),
            "eval_eer_percent": round(float(t1_frozen["eer"] * 100), 3),
            "eval_accuracy_at_frozen_threshold": round(float(t1_frozen["accuracy"] * 100), 2),
            "eval_balanced_accuracy_at_frozen": round(float(t1_frozen["balanced_accuracy"] * 100), 2),
            "eval_roc_auc": round(float(t1_frozen["roc_auc"]), 4),
            "eval_far_fake_miss_rate": round(float(t1_frozen["far"] * 100), 2),
            "eval_frr_human_false_alarm": round(float(t1_frozen["frr"] * 100), 2),
            "eval_confusion_matrix": t1_frozen["confusion_matrix"],
        },
        "per_attack_vulnerability": {
            "hardest_unseen_attacks": t2_df[t2_df["type"] == "Unseen in Eval"].sort_values(by="detection_rate_at_frozen")[["attack_id", "detection_rate_at_frozen", "eer_vs_bonafide"]].head(5).to_dict(orient="records"),
            "seen_attacks_mean_detection": round(float(t2_df[t2_df["type"] == "Seen in Training"]["detection_rate_at_frozen"].mean() * 100), 2),
            "unseen_attacks_mean_detection": round(float(t2_df[t2_df["type"] == "Unseen in Eval"]["detection_rate_at_frozen"].mean() * 100), 2),
        },
        "seed_variance": {
            "mean_dev_accuracy": round(float(t5_df["dev_accuracy"].mean() * 100), 2),
            "std_dev_accuracy": round(float(t5_df["dev_accuracy"].std() * 100), 3),
            "mean_dev_eer": round(float(t5_df["dev_eer"].mean() * 100), 3),
            "std_dev_eer": round(float(t5_df["dev_eer"].std() * 100), 3),
        },
        "robustness_impact": {
            "telephone_simulation_eer": round(float(t6_df[t6_df["Condition"].str.contains("Telephone")]["EER"].values[0] * 100), 2),
            "mp3_32kbps_eer": round(float(t6_df[t6_df["Condition"].str.contains("32")]["EER"].values[0] * 100), 2),
            "noise_5db_eer": round(float(t6_df[t6_df["Condition"].str.contains("5 dB")]["EER"].values[0] * 100), 2),
        },
        "out_of_domain_generators": t7_res,
        "api_performance": t11_res,
        "calibration": {
            "dev_ece_percent": round(float(t9_df[t9_df["Split"] == "Dev"]["ECE"].values[0] * 100), 2),
            "eval_ece_percent": round(float(t9_df[t9_df["Split"] == "Eval"]["ECE"].values[0] * 100), 2),
        },
    }

    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Machine-readable summary written to {summary_file}")

if __name__ == "__main__":
    main()
