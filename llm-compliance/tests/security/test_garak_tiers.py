import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import os
import pytest
import yaml


def _latest_report_file(report_dir: Path, prefix: str) -> Path:
    # garak typically writes: <prefix>.<uuid>.report.jsonl (and .html alongside)
    candidates = sorted(report_dir.glob(f"{prefix}*.report.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No report jsonl found in {report_dir} for prefix={prefix}")
    return candidates[0]


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_eval_scores(report_jsonl: Path):
    """
    Garak JSONL contains rows with entry_type; eval-type entries include the results used to compute score.
    We'll aggregate per (probe, detector) score if present.
    """
    eval_rows = []
    for row in _load_jsonl(report_jsonl):
        if row.get("entry_type") == "eval":
            eval_rows.append(row)
    return eval_rows


@pytest.mark.integration
def test_garak_promptinject_gate():
    """
    CI gate example:
      - Run a small set of probes
      - Fail build if any eval score exceeds threshold (or if any 'hit' count > 0)
    """

    # Get the project root (llm-compliance directory)
    test_dir = Path(__file__).parent  # tests/security/
    project_root = test_dir.parent.parent  # llm-compliance/
    
    # Save reports in llm-compliance/reports/ directory
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = str(reports_dir / f"garak_scan_{timestamp}")  # garak uses this as a prefix for report filenames
    
    # Load generator config from llm-compliance/config/fastapi_rest.json
    config_file = project_root / "config" / "fastapi_rest.json"
    
      
    # Load thresholds from llm-compliance/tests/config/thresholds.yaml
    thresholds_file = test_dir.parent / "config" / "thresholds.yaml"
    assert thresholds_file.exists(), f"Missing thresholds file: {thresholds_file}"
    try:
        with open(thresholds_file, "r", encoding="utf-8") as f:
            thresholds = yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f"Failed to parse thresholds.yaml: {e}")
    except Exception as e:
        pytest.fail(f"Failed to read thresholds.yaml: {e}")

    # Read threshold from environment variable first, then fall back to thresholds.yaml
    
    threshold = thresholds.get("garak", {}).get("eval_threshold", 0.5)  # default to 0.5 if not set
    threshold = float(threshold)

    # Find garak executable (check common locations)
    garak_cmd = os.environ.get("GARAK_CMD", "garak")
    if not garak_cmd or garak_cmd == "garak":
        # Try to find garak in common venv locations
        venv_garak = project_root.parent / ".venv-garak" / "Scripts" / "garak.exe"
        if venv_garak.exists():
            garak_cmd = str(venv_garak)
        else:
            # Fall back to system garak
            garak_cmd = "garak"

    cmd = [
        garak_cmd,
        "--target_type", "rest",
        "-G", str(config_file),
        "--probes", "promptinject",          # change as needed
        "--report_prefix", report_prefix,    # controls output prefix
        "--parallel_attempts", "2",
    ]

    # Prepare environment variables - disable LangSmith tracing to avoid rate limit errors
    env = os.environ.copy()
    env["LANGCHAIN_TRACING_V2"] = "false"  # Disable LangSmith tracing
    # Optionally unset API key to ensure no tracing
    if "LANGCHAIN_API_KEY" in env:
        env.pop("LANGCHAIN_API_KEY", None)
    if "LANGSMITH_API_KEY" in env:
        env.pop("LANGSMITH_API_KEY", None)
    
    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Always save stdout/stderr for debugging
    try:
        (reports_dir / f"garak_stdout_{timestamp}.txt").write_text(run.stdout, encoding="utf-8")
        (reports_dir / f"garak_stderr_{timestamp}.txt").write_text(run.stderr, encoding="utf-8")
    except Exception as e:
        pytest.fail(f"Failed to save garak output files: {e}")

    assert run.returncode == 0, (
        f"garak failed with return code {run.returncode}. "
        f"See artifacts in {reports_dir}. "
        f"stderr: {run.stderr[:500]}"
    )

    # Find the report file (garak may append UUID to the prefix or write to default location)
    report_jsonl = None
    
    # First, try to find in the specified reports directory
    try:
        report_jsonl = _latest_report_file(reports_dir, prefix=f"garak_scan_{timestamp}")
    except FileNotFoundError:
        # If not found, check Garak's default report directory
       
        garak_default_dir = Path.home() / ".local" / "share" / "garak" / "garak_runs"
        if garak_default_dir.exists():
            try:
                # Look for any recent garak report files
                all_reports = sorted(
                    garak_default_dir.glob("garak.*.report.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if all_reports:
                    # Use the most recent report (likely from this run)
                    report_jsonl = all_reports[0]
                    # Copy it to our reports directory for consistency
                    dest_report = reports_dir / report_jsonl.name
                    shutil.copy2(report_jsonl, dest_report)
                    report_jsonl = dest_report
                    print(f"Found report in default location, copied to {dest_report}")
            except Exception as e:
                pass
    
    # If still not found, provide detailed error
    if report_jsonl is None or not report_jsonl.exists():
        error_msg = (
            f"No garak report found with prefix 'garak_scan_{timestamp}'.\n"
            f"Checked locations:\n"
            f"  1. {reports_dir}\n"
            f"  2. {Path.home() / '.local' / 'share' / 'garak' / 'garak_runs'}\n\n"
            f"Possible reasons:\n"
            f"  - Garak failed before writing the report\n"
            f"  - Report was written with a different prefix\n"
            f"  - Check stdout/stderr files in {reports_dir}\n"
            f"  - Check Garak's default location: {Path.home() / '.local' / 'share' / 'garak' / 'garak_runs'}"
        )
        pytest.fail(error_msg)

    eval_rows = _extract_eval_scores(report_jsonl)

    # ---- Assertions / gating rules (customize to your org) ----
    # Rule idea 1: ensure eval rows exist
    assert len(eval_rows) > 0, f"No eval rows found in report: {report_jsonl}"

    # Load tier configuration from thresholds.yaml
    garak_config = thresholds.get("garak", {})
    tiers = {}
    for tier_name in ["tier1", "tier2", "tier3"]:
        tier_config = garak_config.get(tier_name, {})
        if tier_config:
            tiers[tier_name] = {
                "probes": set(tier_config.get("probes", [])),
                "min_pass_rate": float(tier_config.get("min_pass_rate", 1.0))
            }

    # Create a reverse mapping: probe_name -> tier_name
    probe_to_tier = {}
    for tier_name, tier_info in tiers.items():
        for probe in tier_info["probes"]:
            probe_to_tier[probe] = tier_name

    # Group eval rows by tier and probe, calculate pass rates
    tier_results = {}
    probe_results = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "offenders": []})
    probe_eval_rows = defaultdict(list)  # Store all eval rows for each probe
    all_offenders = []
    
    for r in eval_rows:
        probe_name = r.get("probe", "unknown")
        detector_name = r.get("detector", "unknown")
        score = r.get("score")
        
        # Store all eval rows grouped by probe for detailed reporting
        if probe_name:
            probe_eval_rows[probe_name].append(r)
        
        if score is None:
            continue
            
        try:
            score_float = float(score)
            is_failure = score_float >= threshold
            
            # Find which tier this probe belongs to
            tier_name = probe_to_tier.get(probe_name, "untiered")
            
            # Initialize tier result if not exists
            if tier_name not in tier_results:
                tier_results[tier_name] = {
                    "tier": tier_name,
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "pass_rate": 0.0,
                    "min_pass_rate": tiers.get(tier_name, {}).get("min_pass_rate", 1.0) if tier_name != "untiered" else 1.0,
                    "offenders": []
                }
            
            # Update tier statistics
            tier_results[tier_name]["total_tests"] += 1
            if is_failure:
                tier_results[tier_name]["failed_tests"] += 1
                offender = {
                    "probe": probe_name,
                    "detector": detector_name,
                    "score": score_float,
                    "tier": tier_name
                }
                tier_results[tier_name]["offenders"].append(offender)
                all_offenders.append(offender)
            else:
                tier_results[tier_name]["passed_tests"] += 1
            
            # Update probe statistics
            probe_results[probe_name]["total"] += 1
            if is_failure:
                probe_results[probe_name]["failed"] += 1
                probe_results[probe_name]["offenders"].append({
                    "detector": detector_name,
                    "score": score_float,
                    "tier": tier_name
                })
            else:
                probe_results[probe_name]["passed"] += 1
                
        except (ValueError, TypeError):
            # Skip rows with invalid score values
            continue

    # Calculate pass rates for each tier
    tier_failures = []
    for tier_name, tier_data in tier_results.items():
        if tier_data["total_tests"] > 0:
            tier_data["pass_rate"] = tier_data["passed_tests"] / tier_data["total_tests"]
        else:
            tier_data["pass_rate"] = 1.0  # No tests = 100% pass
        
        # Check if tier meets minimum pass rate requirement
        if tier_data["pass_rate"] < tier_data["min_pass_rate"]:
            tier_failures.append({
                "tier": tier_name,
                "pass_rate": tier_data["pass_rate"],
                "min_pass_rate": tier_data["min_pass_rate"],
                "total_tests": tier_data["total_tests"],
                "failed_tests": tier_data["failed_tests"]
            })

    # Save tier-wise CSV report
    tier_csv_path = reports_dir / f"garak_tier_results_{timestamp}.csv"
    try:
        with open(tier_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tier", "total_tests", "passed_tests", "failed_tests", 
                "pass_rate", "min_pass_rate", "status"
            ])
            for tier_name in sorted(tier_results.keys()):
                tier_data = tier_results[tier_name]
                status = "PASS" if tier_data["pass_rate"] >= tier_data["min_pass_rate"] else "FAIL"
                writer.writerow([
                    tier_name,
                    tier_data["total_tests"],
                    tier_data["passed_tests"],
                    tier_data["failed_tests"],
                    f"{tier_data['pass_rate']:.4f}",
                    f"{tier_data['min_pass_rate']:.4f}",
                    status
                ])
    except Exception as e:
        pytest.fail(f"Failed to save tier CSV report: {e}")

    # Save probe-wise CSV report
    probe_csv_path = reports_dir / f"garak_probe_results_{timestamp}.csv"
    try:
        with open(probe_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "probe", "tier", "total_tests", "passed_tests", "failed_tests", 
                "pass_rate", "offender_count"
            ])
            for probe_name in sorted(probe_results.keys()):
                probe_data = probe_results[probe_name]
                tier_name = probe_to_tier.get(probe_name, "untiered")
                pass_rate = probe_data["passed"] / probe_data["total"] if probe_data["total"] > 0 else 1.0
                writer.writerow([
                    probe_name,
                    tier_name,
                    probe_data["total"],
                    probe_data["passed"],
                    probe_data["failed"],
                    f"{pass_rate:.4f}",
                    len(probe_data["offenders"])
                ])
    except Exception as e:
        pytest.fail(f"Failed to save probe CSV report: {e}")

    # Save comprehensive JSON summary for CI
    summary_path = reports_dir / f"garak_summary_{timestamp}.json"
    summary_data = {
        "overall_threshold": threshold,
        "total_eval_rows": len(eval_rows),
        "tier_results": tier_results,
        "tier_failures": tier_failures,
        "probe_results": {
            probe: {
                "total": data["total"],
                "passed": data["passed"],
                "failed": data["failed"],
                "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 1.0,
                "tier": probe_to_tier.get(probe, "untiered"),
                "offender_count": len(data["offenders"])
            }
            for probe, data in probe_results.items()
        },
        "all_offenders": all_offenders
    }
    
    try:
        # Format JSON with proper indentation
        formatted_json = json.dumps(
            summary_data, 
            indent=2, 
            default=str,
            ensure_ascii=False,  # Allow unicode characters
            sort_keys=False  # Maintain order of keys
        )
        summary_path.write_text(formatted_json, encoding="utf-8")
    except Exception as e:
        pytest.fail(f"Failed to save summary file: {e}")

    # Save individual probe details in evidence folder
    evidence_dir = project_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON for each probe
    for probe_name, probe_rows in probe_eval_rows.items():
        # Get probe stats, defaulting to zeros if probe wasn't processed (no valid scores)
        probe_stats = probe_results.get(probe_name, {"total": 0, "passed": 0, "failed": 0, "offenders": []})
        probe_detail_data = {
            "probe": probe_name,
            "tier": probe_to_tier.get(probe_name, "untiered"),
            "timestamp": timestamp,
            "total_tests": len(probe_rows),
            "summary": {
                "total": probe_stats["total"],
                "passed": probe_stats["passed"],
                "failed": probe_stats["failed"],
                "pass_rate": probe_stats["passed"] / probe_stats["total"] if probe_stats["total"] > 0 else 1.0,
                "offender_count": len(probe_stats["offenders"])
            },
            "offenders": probe_stats["offenders"],
            "all_test_results": probe_rows
        }
        
        probe_evidence_path = evidence_dir / f"garak_probe_{probe_name}_{timestamp}.json"
        try:
            # Format JSON with proper indentation and ensure consistent formatting
            formatted_json = json.dumps(
                probe_detail_data, 
                indent=2, 
                default=str,
                ensure_ascii=False,  # Allow unicode characters
                sort_keys=False  # Maintain order of keys
            )
            probe_evidence_path.write_text(formatted_json, encoding="utf-8")
        except Exception as e:
            pytest.fail(f"Failed to save probe evidence file for {probe_name}: {e}")

    # Assert based on tier pass rates
    if tier_failures:
        failure_messages = []
        for failure in tier_failures:
            failure_messages.append(
                f"{failure['tier']}: pass_rate={failure['pass_rate']:.2%} "
                f"(required: {failure['min_pass_rate']:.2%}), "
                f"failed_tests={failure['failed_tests']}/{failure['total_tests']}"
            )
        
        pytest.fail(
            f"Garak gate failed: {len(tier_failures)} tier(s) below minimum pass rate. "
            f"{'; '.join(failure_messages)}. "
            f"Reports saved to {reports_dir}"
        )
    
    # Also check overall threshold for backward compatibility
    if all_offenders:
        pytest.fail(
            f"Garak gate failed: {len(all_offenders)} offense(s) found above threshold {threshold}. "
            f"Reports saved to {reports_dir}"
        )
