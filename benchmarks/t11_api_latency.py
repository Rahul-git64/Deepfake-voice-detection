import os
import sys
import io
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch

from benchmarks.utils import (
    get_wav2vec2,
    get_classifier,
    preprocess_audio_bytes,
    extract_embedding_from_waveform,
    predict_from_embedding,
    FROZEN_DEV_THRESHOLD,
    TARGET_SR,
    DEV_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
)

API_HEALTH_URL = "http://127.0.0.1:8080/health"
API_PREDICT_URL = "http://127.0.0.1:8080/predict"

def run_t11(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T11: API Latency, Throughput & Parity Verification")
    print("=" * 60)

    # 1. Check API Health
    try:
        resp = requests.get(API_HEALTH_URL, timeout=5)
        assert resp.status_code == 200, f"API returned status {resp.status_code}"
        health_info = resp.json()
        print(f"Connected to API: {health_info}")
    except Exception as e:
        print(f"Error connecting to FastAPI backend: {e}")
        print("Please ensure the backend is running at http://127.0.0.1:8080")
        return None

    # Load 50 clips for Parity Testing
    from benchmarks.utils import get_dev_protocol_df
    dev_df = get_dev_protocol_df()
    parity_sample = dev_df.sample(n=10 if quick else 50, random_state=RANDOM_SEED)

    wav2vec2 = get_wav2vec2()
    classifier = get_classifier()

    print("\n--- 1. Parity Check: In-Process Inference vs API /predict ---")
    max_prob_diff = 0.0
    parity_records = []

    for _, row in parity_sample.iterrows():
        fn = row["filename"]
        flac_path = DEV_DIR / f"{fn}.flac"
        if not flac_path.exists():
            continue

        with open(flac_path, "rb") as f:
            audio_bytes = f.read()

        # In-process calculation
        waveform = preprocess_audio_bytes(audio_bytes)
        emb = extract_embedding_from_waveform(waveform, wav2vec2_model=wav2vec2)
        in_proc_probs, in_proc_spoof = predict_from_embedding(emb, classifier=classifier)

        # API call
        files = {"file": (f"{fn}.flac", audio_bytes, "audio/flac")}
        api_resp = requests.post(API_PREDICT_URL, files=files, timeout=10)
        assert api_resp.status_code == 200, f"API error: {api_resp.text}"
        api_data = api_resp.json()
        api_spoof = api_data["probabilities"]["spoof"]

        diff = abs(in_proc_spoof - api_spoof)
        max_prob_diff = max(max_prob_diff, diff)

        parity_records.append({
            "filename": fn,
            "in_process_spoof": in_proc_spoof,
            "api_spoof": api_spoof,
            "diff": diff,
            "parity_pass": diff <= 1e-3,  # rounded to 4 decimals in API response
        })

    parity_df = pd.DataFrame(parity_records)
    all_passed = parity_df["parity_pass"].all()
    print(f"Parity Check (n={len(parity_records)}): Max Abs Difference = {max_prob_diff:.6f} | All Pass: {all_passed}")

    # 2. Sequential Latency Benchmark (100 requests)
    print("\n--- 2. Sequential Latency Benchmark ---")
    benchmark_clip = parity_records[0]["filename"]
    clip_bytes = open(DEV_DIR / f"{benchmark_clip}.flac", "rb").read()

    n_requests = 20 if quick else 100
    latencies_client_ms = []
    latencies_server_ms = []

    for req_i in range(n_requests):
        files = {"file": ("bench.flac", clip_bytes, "audio/flac")}
        t0 = time.time()
        resp = requests.post(API_PREDICT_URL, files=files, timeout=10)
        client_elapsed_ms = (time.time() - t0) * 1000.0

        if resp.status_code == 200:
            data = resp.json()
            latencies_client_ms.append(client_elapsed_ms)
            latencies_server_ms.append(data.get("inference_time_ms", client_elapsed_ms))

    latencies_client_ms = np.array(latencies_client_ms)
    latencies_server_ms = np.array(latencies_server_ms)

    p50 = np.percentile(latencies_client_ms, 50)
    p95 = np.percentile(latencies_client_ms, 95)
    p99 = np.percentile(latencies_client_ms, 99)
    mean_lat = np.mean(latencies_client_ms)
    server_mean = np.mean(latencies_server_ms)

    print(f"Client Latency (n={n_requests}):")
    print(f"  Mean: {mean_lat:.1f} ms | p50: {p50:.1f} ms | p95: {p95:.1f} ms | p99: {p99:.1f} ms")
    print(f"Server-reported Inference Latency Mean: {server_mean:.1f} ms")

    # 3. Concurrency Stress Test
    print("\n--- 3. Concurrency Benchmark (10 concurrent workers, 5 rounds) ---")
    n_workers = 3 if quick else 10
    n_rounds = 2 if quick else 5
    concurrent_latencies = []

    def make_concurrent_request(worker_id):
        files = {"file": (f"worker_{worker_id}.flac", clip_bytes, "audio/flac")}
        t_start = time.time()
        r = requests.post(API_PREDICT_URL, files=files, timeout=30)
        elapsed = (time.time() - t_start) * 1000.0
        return r.status_code == 200, elapsed

    t_concurrent_start = time.time()
    total_concurrent_success = 0
    total_concurrent_reqs = n_workers * n_rounds

    for round_idx in range(n_rounds):
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(make_concurrent_request, w) for w in range(n_workers)]
            for fut in futures:
                ok, lat = fut.result()
                if ok:
                    total_concurrent_success += 1
                concurrent_latencies.append(lat)

    total_concurrent_time = time.time() - t_concurrent_start
    throughput_rps = total_concurrent_success / total_concurrent_time

    print(f"Concurrent Success: {total_concurrent_success}/{total_concurrent_reqs}")
    print(f"Throughput: {throughput_rps:.2f} requests/second")
    print(f"Concurrent Latency Mean: {np.mean(concurrent_latencies):.1f} ms, p95: {np.percentile(concurrent_latencies, 95):.1f} ms")

    # 4. Edge-Case Test Suite
    print("\n--- 4. Edge-Case Validation Suite ---")
    edge_cases = [
        ("Empty file (0 bytes)", b"", "empty.wav", 400),
        ("Renamed text file", b"This is not audio data at all.", "fake_audio.wav", 400),
        ("Truncated MP3 (partial header)", b"\xff\xfb\x90\x00" * 4, "truncated.mp3", 400),
    ]

    # Generate valid long and multi-channel audio
    sr = 16000
    # 60-second audio
    buf_60s = io.BytesIO()
    sf.write(buf_60s, np.zeros(60 * sr, dtype=np.float32), sr, format="WAV")
    edge_cases.append(("60-second audio clip", buf_60s.getvalue(), "60s.wav", 200))

    # Stereo 44.1 kHz audio
    buf_stereo = io.BytesIO()
    stereo_data = np.zeros((44100 * 3, 2), dtype=np.float32)
    sf.write(buf_stereo, stereo_data, 44100, format="WAV")
    edge_cases.append(("Stereo 44.1 kHz audio", buf_stereo.getvalue(), "stereo.wav", 200))

    edge_results = []
    for test_name, data_bytes, fn, expected_status in edge_cases:
        try:
            files = {"file": (fn, data_bytes, "audio/wav")}
            resp = requests.post(API_PREDICT_URL, files=files, timeout=30)
            status = resp.status_code
            passed = (status == expected_status)
            edge_results.append({
                "test": test_name,
                "expected_status": expected_status,
                "actual_status": status,
                "pass": passed,
                "response_preview": resp.text[:100],
            })
            print(f"Edge Case: {test_name:<30} -> Status: {status} (Expected: {expected_status}) -> {'PASS' if passed else 'FAIL'}")
        except Exception as e:
            print(f"Edge Case: {test_name} error: {e}")
            edge_results.append({
                "test": test_name,
                "expected_status": expected_status,
                "actual_status": -1,
                "pass": False,
                "response_preview": str(e),
            })

    # Save summary results
    summary = {
        "parity_max_diff": float(max_prob_diff),
        "parity_passed": bool(all_passed),
        "sequential_p50_ms": float(p50),
        "sequential_p95_ms": float(p95),
        "sequential_p99_ms": float(p99),
        "sequential_mean_ms": float(mean_lat),
        "server_mean_ms": float(server_mean),
        "concurrency_throughput_rps": float(throughput_rps),
        "concurrency_p95_ms": float(np.percentile(concurrent_latencies, 95)),
        "edge_cases_passed": sum(r["pass"] for r in edge_results),
        "edge_cases_total": len(edge_results),
    }

    pd.DataFrame(parity_records).to_csv(RESULTS_DIR / "t11_api_parity.csv", index=False)
    pd.DataFrame(edge_results).to_csv(RESULTS_DIR / "t11_api_edge_cases.csv", index=False)
    pd.Series(summary).to_json(RESULTS_DIR / "t11_api_summary.json")

    print(f"\nAPI benchmark results saved to {RESULTS_DIR}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t11(quick=args.quick, full=args.full)
