import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import requests
import yaml
#probeName = "promptinject"
#probeName = "jailbreak"
probeName = "leakreplay"

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
    
    # Allow override via environment variable
    generator_cfg_env = os.environ.get("GARAK_GEN_CFG", None)
    if generator_cfg_env:
        generator_cfg = Path(generator_cfg_env).resolve()
        generator_cfg_str = str(generator_cfg)
    else:
        # Use relative path from project root (matches command line behavior: config\fastapi_rest.json)
        # When cwd is project_root, this relative path will work
        generator_cfg_str = "config/fastapi_rest.json"
        generator_cfg = project_root / generator_cfg_str
    
    assert generator_cfg.exists(), f"Missing generator config: {generator_cfg}"
    
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
    if "GARAK_EVAL_THRESHOLD" in os.environ:
        try:
            threshold = float(os.environ.get("GARAK_EVAL_THRESHOLD"))
        except ValueError:
            pytest.fail(f"Invalid GARAK_EVAL_THRESHOLD value: {os.environ.get('GARAK_EVAL_THRESHOLD')}")
    else:
        threshold = thresholds.get("garak", {}).get("eval_threshold", 0.5)  # default to 0.5 if not set
        threshold = float(threshold)

    # Check if FastAPI service is running and responsive before running Garak
    api_url = "http://localhost:8000"
    query_endpoint = f"{api_url}/query"
    
    # First, check if service is reachable
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        print(f"Service health check: {response.status_code}")
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"FastAPI service is not running at {api_url}.\n"
            f"Please start the service before running this test:\n"
            f"  python -m app.fastapiragpdfagent\n"
            f"  or\n"
            f"  uvicorn app.fastapiragpdfagent:app --host 0.0.0.0 --port 8000"
        )
    except requests.exceptions.Timeout:
        pytest.skip(
            f"FastAPI service at {api_url} did not respond within 5 seconds.\n"
            f"Service may be hung or overloaded. Please restart the service."
        )
    except Exception as e:
        pytest.skip(f"Could not verify FastAPI service at {api_url}: {e}")
    
    # Test the actual /query endpoint that Garak will use (with a short timeout)
    try:
        test_payload = {"question": "test", "debug": False}
        test_response = requests.post(query_endpoint, json=test_payload, timeout=30)
        print(f"Query endpoint test: {test_response.status_code}")
        if test_response.status_code not in [200, 422]:  # 422 is validation error, but service is working
            pytest.skip(
                f"Query endpoint at {query_endpoint} returned status {test_response.status_code}.\n"
                f"Service may not be configured correctly."
            )
    except requests.exceptions.Timeout:
        pytest.skip(
            f"Query endpoint at {query_endpoint} timed out after 30 seconds.\n"
            f"This indicates the service is hung or taking too long to process requests.\n"
            f"Please restart the FastAPI service and ensure it's responsive.\n"
            f"Check if the RAG agent initialization completed successfully."
        )
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"Cannot connect to query endpoint at {query_endpoint}.\n"
            f"Service may have stopped. Please restart the service."
        )
    except Exception as e:
        pytest.skip(f"Error testing query endpoint: {e}")

    # Use python -m garak (as it works from command line)
    # Find Python executable from venv first, then fall back to system python
    python_cmd = os.environ.get("GARAK_PYTHON", None)
    if not python_cmd:
        # Try to find Python in the garak venv
        venv_python = project_root.parent / ".venv-garak" / "Scripts" / "python.exe"
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            # Fall back to system python
            python_cmd = "python"
    
    # Allow override via environment variable for full command
    garak_cmd_override = os.environ.get("GARAK_CMD", None)
    if garak_cmd_override:
        # If full command is provided, split it (e.g., "python -m garak" or "garak")
        cmd_parts = garak_cmd_override.split()
        cmd = cmd_parts + [
            "--target_type", "rest",
            "-G", generator_cfg_str,
            "--probes", probeName,          # change as needed
            "--report_prefix", report_prefix,    # controls output prefix
            "--parallel_attempts", "1",
            "--generations", "4",
        ]
    else:
        # Use python -m garak (recommended way that works from command line)
        cmd = [
            python_cmd,
            "-m", "garak",
            "--config", "config/garak_PR_config.yaml",
            "--target_type", "rest",
            "-G", generator_cfg_str,
            "--probes", probeName,          # change as needed
            "--report_prefix", report_prefix,    # controls output prefix
            "--parallel_attempts", "1",
    #        "--generations", "4",
        ]

    print(f"Running garak with command: {cmd}")
    print(f"Python command: {python_cmd}")
    print(f"Config file: {generator_cfg}")
    print(f"Report prefix: {report_prefix}")
    print(f"Working directory: {os.getcwd()}")
    
    # Prepare environment variables - disable LangSmith tracing to avoid rate limit errors
    env = os.environ.copy()
    env["LANGCHAIN_TRACING_V2"] = "false"  # Disable LangSmith tracing
    # Optionally unset API key to ensure no tracing
    if "LANGCHAIN_API_KEY" in env:
        env.pop("LANGCHAIN_API_KEY", None)
    if "LANGSMITH_API_KEY" in env:
        env.pop("LANGSMITH_API_KEY", None)
    
    # Use UTF-8 encoding to handle Unicode characters from Garak
    # Set working directory to project root to ensure relative paths work
    run = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='replace',  # Replace problematic characters instead of failing
        cwd=str(project_root),  # Run from project root to match command line behavior
        env=env  # Use modified environment without LangSmith tracing
    )
    print(f"Garak return code: {run.returncode}")
    print(f"Garak stdout (first 500 chars): {run.stdout[:500]}")
    print(f"Garak stderr (first 500 chars): {run.stderr[:500]}")
    # Always save stdout/stderr for debugging
    try:
        (reports_dir / f"garak_stdout_{timestamp}.txt").write_text(run.stdout, encoding="utf-8")
        (reports_dir / f"garak_stderr_{timestamp}.txt").write_text(run.stderr, encoding="utf-8")
    except Exception as e:
        pytest.fail(f"Failed to save garak output files: {e}")

    # Note: Garak may return exit code 1 even when it completes successfully
    # (e.g., when vulnerabilities are found). We should check for the report file
    # first and only fail if there's no report AND a non-zero return code.
    # Check for specific timeout errors first (these are always failures)
    if run.returncode != 0:
        error_msg = run.stderr[:1000] if run.stderr else run.stdout[-1000:] if run.stdout else "Unknown error"
        
        # Check for specific timeout errors - these are always failures
        if "ReadTimeout" in error_msg or "read timeout" in error_msg.lower():
            pytest.fail(
                f"Garak timed out while connecting to the API. "
                f"This usually means:\n"
                f"  1. The FastAPI service at http://localhost:8000 is not running\n"
                f"  2. The service is taking too long to respond (>300s)\n"
                f"  3. The service may be hung or overloaded\n\n"
                f"Please ensure the FastAPI service is running and responsive before running this test.\n"
                f"See artifacts in {reports_dir}.\n"
                f"Error: {error_msg}"
            )
        # For other non-zero return codes, we'll check if a report was generated
        # If a report exists, we'll continue processing it (Garak may return 1 for found vulnerabilities)
        # If no report exists, then it's a real failure
        print(f"Warning: Garak returned exit code {run.returncode}. Checking for report file...")

    # Find the report file (garak may append UUID to the prefix or write to default location)
    report_jsonl = None
    
    # First, try to find in the specified reports directory with expected prefix
    try:
        report_jsonl = _latest_report_file(reports_dir, prefix=f"garak_scan_{timestamp}")
        print(f"Found report with prefix: {report_jsonl}")
    except FileNotFoundError:
        # If not found with prefix, try to find any recent garak report files in reports_dir
        try:
            all_reports = sorted(
                reports_dir.glob("garak*.report.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if all_reports:
                # Use the most recent report (likely from this run)
                report_jsonl = all_reports[0]
                print(f"Found report without prefix match: {report_jsonl.name}")
        except Exception:
            pass
        
        # If still not found, check Garak's default report directory
        if report_jsonl is None or not report_jsonl.exists():
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
                    print(f"Error checking default location: {e}")
    
    # If still not found, provide detailed error
    # If Garak returned non-zero and no report exists, it's a real failure
    if report_jsonl is None or not report_jsonl.exists():
        error_msg = (
            f"No garak report found with prefix 'garak_scan_{timestamp}'.\n"
            f"Checked locations:\n"
            f"  1. {reports_dir} (with prefix 'garak_scan_{timestamp}')\n"
            f"  2. {reports_dir} (any garak*.report.jsonl files)\n"
            f"  3. {Path.home() / '.local' / 'share' / 'garak' / 'garak_runs'}\n\n"
            f"Possible reasons:\n"
            f"  - Garak failed before writing the report\n"
            f"  - Report was written with a different prefix/name\n"
            f"  - Check stdout/stderr files in {reports_dir}\n"
            f"  - Check Garak's default location: {Path.home() / '.local' / 'share' / 'garak' / 'garak_runs'}\n"
            f"  - Garak may ignore --report_prefix and use its own naming"
        )
        # If Garak returned non-zero exit code and no report exists, it's a real failure
        if run.returncode != 0:
            error_msg += f"\n\nGarak returned exit code {run.returncode} and no report was generated."
            error_msg += f"\nStderr: {run.stderr[:500] if run.stderr else 'None'}"
        pytest.fail(error_msg, pytrace=False)

    eval_rows = _extract_eval_scores(report_jsonl)

    # ---- Assertions / gating rules (customize to your org) ----
    # Rule idea 1: ensure eval rows exist
    assert len(eval_rows) > 0, f"No eval rows found in report: {report_jsonl}"

    # Rule idea 2: fail if any eval score >= threshold
    # Note: field names can vary by version/plugins; check row keys in your report.
    offenders = []
    for r in eval_rows:
        score = r.get("score")
        if score is not None:
            try:
                score_float = float(score)
                if score_float >= threshold:
                    offenders.append({
                        "probe": r.get("probe", "unknown"),
                        "detector": r.get("detector", "unknown"),
                        "score": score_float,
                    })
            except (ValueError, TypeError):
                # Skip rows with invalid score values
                continue

    # Get threshold for the specific probe from thresholds.yaml
    garak_config = thresholds.get("garak", {})
    min_pass_rate = None
    tier_name = None
    
    # Search through tiers to find the probe and its min_pass_rate
    for tier_key in ['tier1', 'tier2', 'tier3']:
        tier_config = garak_config.get(tier_key, {})
        tier_probes = tier_config.get('probes', [])
        
        # Check if probeName matches any probe in this tier
        if isinstance(tier_probes, list):
            for probe_item in tier_probes:
                # Handle both string and dict probe items
                probe_name = probe_item if isinstance(probe_item, str) else list(probe_item.keys())[0] if isinstance(probe_item, dict) else None
                
                if probe_name and (probe_name == probeName or probe_name.startswith(probeName) or probeName.startswith(probe_name)):
                    # Found the probe in this tier, use tier's min_pass_rate
                    min_pass_rate = tier_config.get('min_pass_rate')
                    tier_name = tier_key
                    break
        
        if min_pass_rate is not None:
            break
    
    # Fallback to default threshold if probe not found in tiers
    if min_pass_rate is None:
        min_pass_rate = threshold
        tier_name = "default"
        print(f"Warning: Probe '{probeName}' not found in thresholds.yaml tiers, using default threshold: {min_pass_rate}")
    else:
        print(f"Found probe '{probeName}' in {tier_name} with min_pass_rate: {min_pass_rate}")

    # Filter eval rows for the specific probe
    probe_eval_rows = [r for r in eval_rows if r.get("probe", "").startswith(probeName)]
    
    if not probe_eval_rows:
        pytest.fail(f"No eval rows found for probe '{probeName}' in report: {report_jsonl}")

    # Calculate overall pass rate for this probe (like the example)
    total_tests = sum(r.get('total', 0) for r in probe_eval_rows)
    total_passed = sum(r.get('passed', 0) for r in probe_eval_rows)
    pass_rate = total_passed / total_tests if total_tests > 0 else 0

    # Save a small JSON summary for CI
    summary_path = reports_dir / f"garak_summary_{timestamp}.json"
    try:
        summary_path.write_text(
            json.dumps({
                "probe": probeName,
                "tier": tier_name,
                "total_tests": total_tests,
                "total_passed": total_passed,
                "pass_rate": pass_rate,
                "min_pass_rate": min_pass_rate,
                "threshold": threshold,
                "offenders": offenders,
                "total_eval_rows": len(eval_rows),
                "probe_eval_rows": len(probe_eval_rows)
            }, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        pytest.fail(f"Failed to save summary file: {e}")

    # Assert: Check for offenders (score-based check)
    assert not offenders, (
        f"Garak gate failed: {len(offenders)} offense(s) found with score >= threshold. "
        f"Threshold: {threshold}. "
        f"Offenders saved to {summary_path}"
    )

    # Assert: Check minimum pass rate for the specific probe (like the example)
    assert pass_rate >= min_pass_rate, (
        f"Probe '{probeName}' (tier: {tier_name}) pass rate {pass_rate:.2%} "
        f"below minimum {min_pass_rate:.2%} ({total_passed}/{total_tests} passed). "
        f"Results saved to {summary_path}"
    )
