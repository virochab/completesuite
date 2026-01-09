import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Accept file path as command-line argument or use default
if len(sys.argv) > 1:
    report_file = Path(sys.argv[1])
else:
    # Default to the most recent garak report in the default location
    garak_default_dir = Path.home() / ".local" / "share" / "garak" / "garak_runs"
    if garak_default_dir.exists():
        all_reports = sorted(
            garak_default_dir.glob("garak.*.report.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if all_reports:
            report_file = all_reports[0]
        else:
            print("No garak report files found in default location")
            sys.exit(1)
    else:
        print(f"Garak default directory not found: {garak_default_dir}")
        print("Usage: python filter_garaklog.py <path_to_report.jsonl>")
        sys.exit(1)

# Remove file:/// URI scheme if present
report_path_str = str(report_file)
if report_path_str.startswith("file:///"):
    # Convert file:///C:/path to C:/path
    report_path_str = report_path_str[8:]  # Remove "file:///"
elif report_path_str.startswith("file://"):
    # Convert file:///path to /path (Unix)
    report_path_str = report_path_str[7:]  # Remove "file://"

report_file = Path(report_path_str)

if not report_file.exists():
    print(f"Report file not found: {report_file}")
    sys.exit(1)

failures = []
passed_attempts = []  # Track attempts that passed (no detector flags)
eval_failures = []
failed_probes = set()  # Track which probes have failures
threshold = 0.5  # Default threshold (50%), can be overridden via command line

# Parse threshold from command line if provided
if len(sys.argv) > 2:
    try:
        threshold = float(sys.argv[2])
    except ValueError:
        print(f"Warning: Invalid threshold '{sys.argv[2]}', using default 0.5")

print(f"Reading report file: {report_file}")
print(f"Using threshold: {threshold} (scores >= {threshold*100}% are considered failures)")
print()

# First pass: collect eval failures to know which probes failed
failing_probe_names = set()

# First, read eval entries to identify failing probes
with open(report_file, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("entry_type") == "eval":
            passed = row.get("passed")
            total = row.get("total")
            if passed is not None and total is not None and total > 0:
                try:
                    score_float = float(passed) / float(total)
                    if score_float >= threshold:
                        probe_name = row.get("probe", "unknown")
                        failing_probe_names.add(probe_name)
                        eval_failures.append({
                            "probe": probe_name,
                            "detector": row.get("detector", "unknown"),
                            "score": score_float,
                            "score_percent": f"{score_float*100:.1f}%",
                            "passed": int(passed),
                            "total": int(total),
                            "failed": int(total) - int(passed)
                        })
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

# Second pass: collect attempt-level failures
with open(report_file, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        entry_type = row.get("entry_type")
        
        # Check attempt-level failures
        if entry_type == "attempt":
            # Extract prompt text from nested structure
            prompt_text = None
            if "prompt" in row and "turns" in row["prompt"]:
                if len(row["prompt"]["turns"]) > 0:
                    prompt_text = row["prompt"]["turns"][0].get("content", {}).get("text", "")
            
            # Extract response text from outputs array (use first output)
            response_text = None
            if "outputs" in row and len(row["outputs"]) > 0:
                response_text = row["outputs"][0].get("text", "")
            
            # Check detector_results - detector_results structure:
            # A dict with detector_name: [list of scores] where each score is 0.0 or 1.0
            # Score of 1.0 means the detector detected malicious content in that output
            detector_results = row.get("detector_results", {})
            probe_name = row.get("probe_classname", "unknown")
            
            # Check if any detector flagged this as a failure
            is_failure = False
            failing_detectors = []
            for detector_name, result in detector_results.items():
                # Handle both formats: single value or list of values
                if isinstance(result, list):
                    # If it's a list, check if any value in the list is 1.0 (detected)
                    # This represents detection across multiple outputs
                    if any(float(v) == 1.0 for v in result if v is not None):
                        is_failure = True
                        failing_detectors.append(detector_name)
                elif isinstance(result, (int, float)):
                    # Single value: 1 or 1.0 means detected/failed, 0 or 0.0 means passed
                    if result == 1 or result == 1.0:
                        is_failure = True
                        failing_detectors.append(detector_name)
            
            # Track both failures and passed attempts
            if is_failure:
                failed_probes.add(probe_name)
                failures.append({
                    "probe": probe_name,
                    "detectors": failing_detectors,
                    "prompt": prompt_text,
                    "response": response_text,
                    "goal": row.get("goal", ""),
                    "all_outputs": [out.get("text", "") for out in row.get("outputs", [])],
                    "has_detector_result": True
                })
            else:
                # This attempt passed (no detector flagged it)
                # Only include if detector_results exists (to avoid including attempts without detectors)
                if detector_results:
                    passed_attempts.append({
                        "probe": probe_name,
                        "detectors": list(detector_results.keys()),
                        "prompt": prompt_text,
                        "response": response_text,
                        "goal": row.get("goal", ""),
                        "all_outputs": [out.get("text", "") for out in row.get("outputs", [])],
                        "has_detector_result": False
                    })

print("=" * 60)
print("ATTEMPT-LEVEL FAILURES (individual prompt/response pairs)")
print("=" * 60)
print(f"Total failing attempts: {len(failures)}")
print(f"Total passed attempts: {len(passed_attempts)}")
print(f"Total passed attempts: {len(passed_attempts)}")
if failures:
    print(f"\nFailed probes: {', '.join(sorted(failed_probes))}")
    print("\n" + "=" * 60)
    print("DETAILED FAILURES (showing prompts and responses):")
    print("=" * 60)
    for i, failure in enumerate(failures[:20], 1):  # Show first 20
        print(f"\n--- Failure #{i} ---")
        print(f"Probe: {failure['probe']}")
        print(f"Failing Detectors: {', '.join(failure['detectors'])}")
        if failure.get('goal'):
            print(f"Goal: {failure['goal']}")
        print(f"Prompt: {failure['prompt']}")
        print(f"Response: {failure['response']}")
        if len(failure.get('all_outputs', [])) > 1:
            print(f"\nAll {len(failure['all_outputs'])} generated responses:")
            for j, output in enumerate(failure['all_outputs'], 1):
                print(f"  {j}. {output[:200]}{'...' if len(output) > 200 else ''}")
        print("-" * 60)
    if len(failures) > 20:
        print(f"\n... and {len(failures) - 20} more failures (total: {len(failures)})")

print()
print("=" * 60)
print(f"EVAL-LEVEL FAILURES (aggregated scores >= {threshold*100}%)")
print("=" * 60)
print(f"Total failing eval entries: {len(eval_failures)}")
if eval_failures:
    print("\nFailing probes (sorted by score, highest first):")
    # Sort by score descending
    eval_failures_sorted = sorted(eval_failures, key=lambda x: x["score"], reverse=True)
    for i, failure in enumerate(eval_failures_sorted, 1):
        print(f"{i}. Probe: {failure['probe']}, Detector: {failure['detector']}, "
              f"Score: {failure['score_percent']} ({failure['passed']}/{failure['total']} detected)")
else:
    print("No eval failures found (all scores below threshold)")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Attempt-level failures: {len(failures)}")
print(f"Eval-level failures (score >= {threshold*100}%): {len(eval_failures)}")
if eval_failures:
    print(f"\n⚠️  WARNING: {len(eval_failures)} probe(s) have scores above threshold!")
    print("   These indicate potential security vulnerabilities.")

# Save results to CSV files in llm-compliance/reports/ directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Determine project root (llm-compliance directory)
# If report_file is in reports/, use that; otherwise try to find llm-compliance root
if "reports" in str(report_file.parent):
    # Report is in a reports directory, go up to find llm-compliance root
    current = report_file.parent
    while current.name != "llm-compliance" and current.parent != current:
        current = current.parent
    if current.name == "llm-compliance":
        output_dir = current / "reports"
    else:
        # Fallback: use reports directory relative to script location
        script_dir = Path(__file__).parent  # tests/security/
        output_dir = script_dir.parent.parent / "reports"  # llm-compliance/reports/
else:
    # Fallback: use reports directory relative to script location
    script_dir = Path(__file__).parent  # tests/security/
    output_dir = script_dir.parent.parent / "reports"  # llm-compliance/reports/

# Create reports directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Save attempt-level failures to CSV
if failures:
    attempts_csv = output_dir / f"garak_attempt_failures_{timestamp}.csv"
    with open(attempts_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['probe', 'detectors', 'goal', 'prompt', 'response', 'all_outputs_count', 'has_detector_result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for failure in failures:
            row = {
                'probe': failure.get('probe', ''),
                'detectors': ', '.join(failure.get('detectors', [])),
                'goal': failure.get('goal', ''),
                'prompt': failure.get('prompt', ''),
                'response': failure.get('response', ''),
                'all_outputs_count': len(failure.get('all_outputs', [])),
                'has_detector_result': failure.get('has_detector_result', False)
            }
            writer.writerow(row)
    print(f"\n✅ Attempt-level failures saved to: {attempts_csv}")

# Save attempt-level passed attempts to CSV
if passed_attempts:
    passed_csv = output_dir / f"garak_attempt_passed_{timestamp}.csv"
    with open(passed_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['probe', 'detectors', 'goal', 'prompt', 'response', 'all_outputs_count', 'has_detector_result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for attempt in passed_attempts:
            row = {
                'probe': attempt.get('probe', ''),
                'detectors': ', '.join(attempt.get('detectors', [])),
                'goal': attempt.get('goal', ''),
                'prompt': attempt.get('prompt', ''),
                'response': attempt.get('response', ''),
                'all_outputs_count': len(attempt.get('all_outputs', [])),
                'has_detector_result': attempt.get('has_detector_result', False)
            }
            writer.writerow(row)
    print(f"✅ Attempt-level passed attempts saved to: {passed_csv}")

# Save eval-level failures to CSV
if eval_failures:
    eval_csv = output_dir / f"garak_eval_failures_{timestamp}.csv"
    with open(eval_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['probe', 'detector', 'score', 'score_percent', 'passed', 'total', 'failed', 'pass_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by score descending
        eval_failures_sorted = sorted(eval_failures, key=lambda x: x["score"], reverse=True)
        for failure in eval_failures_sorted:
            row = {
                'probe': failure.get('probe', ''),
                'detector': failure.get('detector', ''),
                'score': failure.get('score', 0),
                'score_percent': failure.get('score_percent', ''),
                'passed': failure.get('passed', 0),
                'total': failure.get('total', 0),
                'failed': failure.get('failed', 0),
                'pass_rate': f"{(failure.get('passed', 0) / failure.get('total', 1) * 100):.1f}%" if failure.get('total', 0) > 0 else '0%'
            }
            writer.writerow(row)
    print(f"✅ Eval-level failures saved to: {eval_csv}")

if not failures and not eval_failures and not passed_attempts:
    print("\nℹ️  No attempts found, no CSV files generated.")
elif not failures and not eval_failures:
    print(f"\nℹ️  No failures found. {len(passed_attempts)} passed attempts saved to CSV.")
