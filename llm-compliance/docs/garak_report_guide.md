# How to Understand Garak HTML Reports

## Overview
Garak HTML reports provide a comprehensive view of security testing results for your LLM system. This guide explains how to read and interpret these reports.

## Report Structure

### 1. **Header Section**
- **Garak Version**: Version of Garak used for the scan
- **Target Generator**: The model/system being tested (e.g., REST API endpoint)
- **Run Start Time**: When the evaluation began
- **Probe Specifications**: Which probes were used (e.g., promptinject, jailbreak, etc.)

### 2. **Summary Statistics**
Look for summary metrics at the top:
- **Total Probes**: Number of different attack probes tested
- **Total Attempts**: Total number of test cases executed
- **Hits/Detections**: Number of vulnerabilities found
- **Pass Rate**: Percentage of tests that passed

### 3. **Main Results Table**

The report typically contains a table with the following columns:

#### **Probe Name**
- The specific attack/technique tested (e.g., `promptinject`, `jailbreak`, `toxicity`)
- Click on probe names to see detailed results

#### **Detector**
- The evaluation method used to assess the response
- Common detectors:
  - `ModelAsJudge`: Uses an LLM to evaluate responses
  - `always.Reject`: Rule-based detector
  - `toxicity`: Toxicity detection
  - `leakre`: Leakage detection

#### **Score**
- Vulnerability score (typically 0.0 to 1.0)
- **Lower scores = Better** (less vulnerable)
- **Higher scores = Worse** (more vulnerable)
- Scores ≥ threshold (usually 0.5) indicate failures

#### **Status**
- **PASS**: Model handled the probe correctly
- **FAIL**: Model was vulnerable to the probe
- **SKIP**: Test was skipped

### 4. **Detailed Results**

Click on individual probe/detector combinations to see:
- **Prompt Used**: The exact prompt that was tested
- **Model Response**: What your model returned
- **Evaluation Details**: Why it passed/failed
- **Metadata**: Additional context about the test

## Color Coding

Garak reports typically use color coding:
- 🟢 **Green**: Passed tests (no vulnerability detected)
- 🔴 **Red**: Failed tests (vulnerability detected)
- 🟡 **Yellow/Orange**: Skipped or warning tests

## Key Metrics to Look For

### 1. **Overall Pass Rate**
- Percentage of tests that passed
- Compare against your tier thresholds:
  - **Tier 1**: 100% pass rate required
  - **Tier 2**: 99% pass rate required
  - **Tier 3**: 95% pass rate required

### 2. **Tier-Wise Breakdown**
Check how your model performs across different security tiers:
- **Tier 1 (Critical)**: promptinject, jailbreak
- **Tier 2 (High)**: toxicity, malwaregen, privacy
- **Tier 3 (Medium)**: bias, hallucination, quality

### 3. **Probe-Specific Results**
- Identify which probes consistently fail
- Look for patterns in failures
- Check if failures are detector-specific

## Interpreting Scores

### Score Ranges
- **0.0 - 0.3**: Excellent (very secure)
- **0.3 - 0.5**: Good (mostly secure)
- **0.5 - 0.7**: Fair (some vulnerabilities)
- **0.7 - 1.0**: Poor (significant vulnerabilities)

### Threshold Comparison
- Your test uses `eval_threshold: 0.5` from `thresholds.yaml`
- Any score ≥ 0.5 is considered a failure
- Scores are aggregated per probe/detector combination

## Common Sections in Garak Reports

### 1. **Probe Results**
- Lists all probes tested
- Shows pass/fail status for each
- Provides aggregate scores

### 2. **Detector Results**
- Shows how different detectors evaluated responses
- Helps identify false positives/negatives

### 3. **Taxonomy View** (if enabled)
- Groups probes by category
- Helps identify areas of weakness

### 4. **Timeline/Chronology**
- Shows when each test was run
- Useful for tracking changes over time

## How to Use the Report

### Step 1: Check Overall Status
1. Look at the summary statistics
2. Check if overall pass rate meets tier requirements
3. Identify total number of failures

### Step 2: Review Tier Performance
1. Check Tier 1 results (must be 100% pass)
2. Review Tier 2 results (must be ≥99% pass)
3. Examine Tier 3 results (must be ≥95% pass)

### Step 3: Investigate Failures
1. Click on failed probes to see details
2. Review the prompts that caused failures
3. Check model responses to understand why it failed
4. Look for patterns (e.g., all failures in one detector)

### Step 4: Take Action
1. **For Tier 1 failures**: Critical - fix immediately
2. **For Tier 2 failures**: High priority - address soon
3. **For Tier 3 failures**: Medium priority - plan fixes

## Tips for Reading Reports

1. **Start with Summary**: Always check the summary first for quick overview
2. **Filter by Tier**: Use filters to focus on specific tiers
3. **Sort by Score**: Sort by vulnerability score to see worst offenders first
4. **Export Data**: Use the JSON/CSV exports for detailed analysis
5. **Compare Runs**: Compare multiple reports to track improvements

## Related Files

When you run Garak tests, you also get:
- **JSONL Report**: Machine-readable detailed results (`*.report.jsonl`)
- **CSV Reports**: Tier and probe-wise summaries (in `reports/` folder)
- **JSON Evidence**: Individual probe details (in `evidence/` folder)

## Example Interpretation

If you see:
- **Probe**: `promptinject`
- **Detector**: `ModelAsJudge`
- **Score**: 0.75
- **Status**: FAIL

This means:
- The prompt injection attack was tested
- An LLM judge evaluated the response
- Score of 0.75 indicates high vulnerability (above 0.5 threshold)
- The model failed to defend against this attack

## Need Help?

- Check the JSON/CSV exports for machine-readable data
- Review individual probe evidence files in `evidence/` folder
- Compare with tier thresholds in `tests/config/thresholds.yaml`
- Consult Garak documentation: https://reference.garak.ai/

