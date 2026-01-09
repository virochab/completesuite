# Jenkins Pipeline Setup Guide

## Overview

This Jenkinsfile provides a comprehensive CI/CD pipeline for running LLM compliance tests with:
- **Scheduled runs**: Daily sanity tests and weekly comprehensive tests
- **Parameterized builds**: Run specific test categories on-demand
- **Artifact archiving**: Automatic collection of test reports and evidence

## Test Categories

### Available Test Types

1. **sanity** - Quick daily validation tests (unit, security basics, privacy, safety)
2. **all** - Comprehensive test suite (all tests except performance)
3. **bias** - Counterfactual bias detection tests
4. **aiquality** - AI quality and behaviour evaluation tests
5. **rag** - RAG quality metrics (RAGAS tests)
6. **security** - Security tests (Garak, authz, rate limiting, tool allowlist)
7. **safety** - Safety compliance tests (toxicity, misuse, role violation, PII)
8. **privacy** - Privacy tests (PII detection, right to erasure, logs/traces)
9. **agentic** - Agentic system tests (trajectory, tool calls, multi-agent)
10. **multiturn** - Multi-turn conversation tests
11. **mcp** - MCP (Model Context Protocol) tests
12. **ops** - Operations tests (fallbacks, latency budget)
13. **unit** - Unit tests
14. **performance** - Performance/load tests (Locust)

## Setup Instructions

### 1. Create Jenkins Pipeline Job

1. In Jenkins, go to **New Item**
2. Select **Pipeline**
3. Enter job name (e.g., `llm-compliance-tests`)
4. Click **OK**

### 2. Configure Pipeline

1. In the job configuration, scroll to **Pipeline** section
2. Select **Pipeline script from SCM**
3. Choose your SCM (Git)
4. Enter repository URL
5. Set **Script Path** to `Jenkinsfile`
6. Click **Save**

### 3. Set Up Scheduled Runs

#### Option A: Using Jenkins UI

1. In job configuration, go to **Build Triggers**
2. Check **Build periodically**
3. Add cron expressions:
   - **Daily Sanity**: `H 2 * * *` (runs daily at 2 AM)
   - **Weekly Comprehensive**: `H 3 * * 0` (runs Sunday at 3 AM)

#### Option B: Using Jenkinsfile Properties

Add this to your Jenkinsfile (after the `options` block):

```groovy
properties([
    pipelineTriggers([
        cron('H 2 * * *'),  // Daily sanity at 2 AM
        cron('H 3 * * 0')   // Weekly comprehensive on Sunday at 3 AM
    ])
])
```

### 4. Environment Variables

Ensure these environment variables are set in Jenkins:

#### For Standard OpenAI:
- `OPENAI_API_KEY` - OpenAI API key for LLM tests
- `PYTHON_VERSION` - Python version (default: 3.11)

#### For Azure OpenAI:
If using Azure OpenAI instead of standard OpenAI, set these variables:

- `AZURE_OPENAI_API_KEY` - Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint URL (e.g., `https://your-resource.openai.azure.com/`)
- `AZURE_OPENAI_API_VERSION` - API version (optional, e.g., `2024-02-15-preview`)
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Deployment name (optional, may be required for some models)

**Note:** If using Azure OpenAI, you may also need to set `OPENAI_API_KEY` to the Azure key value and configure the base URL in your application code, as some libraries use `OPENAI_API_KEY` with a custom base URL for Azure.

#### Setting Environment Variables:

**Option 1: Global Environment Variables**
- **Manage Jenkins** → **Configure System** → **Global properties** → **Environment variables**
- Add variables as key-value pairs

**Option 2: Job-Level Environment Variables**
- In pipeline job configuration → **Build Environment** → **Use secret text(s) or file(s)**
- Add variables using Jenkins credentials binding

**Option 3: Using Jenkins Credentials (Recommended for API Keys)**
1. **Manage Jenkins** → **Credentials** → **System** → **Global credentials** → **Add Credentials**
2. Select **Secret text**
3. Enter your API key as the secret
4. Set ID (e.g., `azure-openai-api-key`)
5. In Jenkinsfile, use: `withCredentials([string(credentialsId: 'azure-openai-api-key', variable: 'AZURE_OPENAI_API_KEY')])`

### 5. Virtual Environments

The pipeline uses **three separate virtual environments** to avoid dependency conflicts:

1. **`.venv`** (main) - For most tests (RAG, bias, aiquality, privacy, agentic, multiturn, mcp, ops, unit)
2. **`.venv-garak`** - For Garak security tests (test_garak.py, test_garak_tiers.py)
3. **`.venv-giskard`** - For Giskard safety tests (test_giskard_safety.py, test_rag_giskard_safety.py)

The pipeline automatically:
- Creates these environments if they don't exist
- Installs appropriate dependencies in each
- Uses the correct environment for each test type

**Note:** If you have pre-existing virtual environments, the pipeline will use them. Otherwise, it will create new ones.

### 6. Required Jenkins Plugins

Install these plugins:
- **Pipeline** (usually pre-installed)
- **JUnit** (for test result reporting)
- **AnsiColor** (for colored console output)
- **Timestamper** (for build timestamps)

## Usage

### Scheduled Runs

- **Daily Sanity**: Runs automatically every day at 2 AM
  - Executes quick validation tests
  - Takes ~10-15 minutes
  
- **Weekly Comprehensive**: Runs automatically every Sunday at 3 AM
  - Executes full test suite
  - Takes ~2-4 hours

### Manual/Parameterized Runs

1. Click **Build with Parameters**
2. Select test type from dropdown
3. Configure options:
   - **GENERATE_REPORTS**: Generate metrics summary (default: true)
   - **ARCHIVE_ARTIFACTS**: Archive test results (default: true)
4. Click **Build**

### Example: Run Security Tests Only

1. Build with Parameters
2. Select `security` from TEST_TYPE
3. Build

This will run:
- **With main venv**: `test_authz.py`, `test_rate_limit.py`, `test_tool_allowlist.py`
- **With .venv-garak**: `test_garak.py`, `test_garak_tiers.py`

The pipeline automatically switches virtual environments for Garak tests.

## Test Execution Details

### Sanity Run (Daily)
- Unit tests
- Basic security tests (rate limit, authz)
- Privacy tests (PII output)
- Safety tests (safety compliance)
- Excludes slow tests (`-m "not slow"`)

### Comprehensive Run (Weekly)
- All test categories
- Full test suite
- Excludes performance tests (require manual server setup)

### Individual Test Categories

Each category runs its specific test files:
- **bias**: `test_deepeval_counterfactual_bias.py`
- **aiquality**: `test_deepeval_behaviour.py`
- **rag**: `test_ragas_rag.py`
- **security**: All files in `tests/security/`
- **safety**: All files in `tests/safety/`
- **privacy**: All files in `tests/privacy/`
- **agentic**: All files in `tests/agentic/`
- **multiturn**: All files in `tests/multiturn/`
- **mcp**: `test_deepeval_mcp.py`
- **ops**: All files in `tests/ops/`
- **unit**: All files in `tests/unit/`

## Artifacts

The pipeline automatically archives:

- **Evidence files**: `evidence/**` (JSONL logs, test evidence)
- **Test reports**: 
  - `reports/**/*.csv` (CSV reports)
  - `reports/**/*.json` (JSON reports)
  - `reports/**/*.html` (HTML reports)
  - `reports/**/*.jsonl` (JSONL reports)
- **Metrics**:
  - `reports/metrics_history.csv`
  - `reports/metrics_summary.json`
- **JUnit XML**: `evidence/junit-*.xml` (for test result visualization)

## Build Timeouts

- Default timeout: **4 hours**
- Adjust in Jenkinsfile `options` block if needed

## Troubleshooting

### Tests Fail to Run

1. Check Python version compatibility
2. Verify all dependencies in `requirements.txt` are installed
3. Check environment variables (especially `OPENAI_API_KEY`)
4. Review build logs for specific error messages

### Scheduled Runs Not Triggering

1. Verify cron expressions in Jenkins job configuration
2. Check Jenkins system time/timezone
3. Ensure job is not disabled
4. Check Jenkins logs for trigger issues

### Performance Tests Not Running

Performance tests require:
- FastAPI server running on `http://localhost:8000`
- Locust installed (`pip install locust`)
- Manual server setup before running tests

To run performance tests:
1. Start FastAPI server: `python -m uvicorn app.fastapiragpdfagent:app --host 0.0.0.0 --port 8000`
2. Run Jenkins build with `TEST_TYPE=performance`

### Missing Artifacts

1. Check if `ARCHIVE_ARTIFACTS` parameter is enabled
2. Verify file paths match expected locations
3. Check disk space on Jenkins server
4. Review build logs for archive errors

## Customization

### Add New Test Category

1. Add choice to `TEST_TYPE` parameter in Jenkinsfile
2. Add command mapping in `getTestCommand()` function
3. Update this documentation

### Modify Scheduled Times

Edit cron expressions:
- Format: `H M D M W`
- Example: `H 4 * * *` = 4 AM daily
- Example: `H 5 * * 1` = 5 AM every Monday

### Adjust Timeout

Modify in `options` block:
```groovy
timeout(time: 6, unit: 'HOURS')  // 6 hours instead of 4
```

## Best Practices

1. **Monitor scheduled runs**: Check daily sanity runs for quick feedback
2. **Use parameterized builds**: Run specific categories for faster iteration
3. **Review artifacts**: Regularly check archived reports for trends
4. **Set up notifications**: Configure email/Slack notifications for failures
5. **Keep dependencies updated**: Regularly update `requirements.txt`

## Support

For issues or questions:
1. Check build logs in Jenkins
2. Review test output in console
3. Examine archived artifacts for details
4. Consult test-specific documentation in `docs/` folder

