# Garak Configuration Guide: `generations` vs `soft_probe_prompt_cap`

This document explains the difference between two important Garak configuration parameters that control how many API calls are made during testing.

## Overview

Both `generations` and `soft_probe_prompt_cap` affect the number of API calls, but they control different aspects:

- **`generations`**: How many times to run the **same prompt** (for statistical confidence)
- **`soft_probe_prompt_cap`**: How many **different prompts** a probe can generate (for limiting test scope)

## `generations` Parameter

### Purpose
Controls how many times each prompt is sent to the model to get multiple responses.

### Effect
Increases statistical confidence by getting multiple responses to the same prompt.

### Example
If a probe generates prompt "A" and `generations: 4`:
- The model will receive prompt "A" **4 times**
- You get **4 responses** to the same prompt
- Useful for testing consistency and variability

### Configuration
```yaml
run:
  generations: 4  # Each prompt will be sent 4 times
```

### Total API Calls
- **Formula**: `1 prompt × 4 generations = 4 API calls`

---

## `soft_probe_prompt_cap` Parameter

### Purpose
Limits how many different prompts/attempts a probe can generate.

### Effect
Reduces the number of unique test cases per probe.

### Important Notes
- **"Soft" means advisory**: Some probes may ignore this limit
- Probes may have minimums that override the cap
- Default value is typically 256 if not specified

### Example
If a probe would normally generate 256 different prompts:
- With `soft_probe_prompt_cap: 5`, it tries to limit to **5 different prompts**
- However, the probe may still generate more if it has a minimum requirement

### Configuration
```yaml
run:
  soft_probe_prompt_cap: 5  # Try to limit to 5 different prompts per probe
```

### Total API Calls
- **Formula**: `5 different prompts × 1 generation = 5 API calls` (if cap is respected)

---

## Combined Example

### Configuration
```yaml
run:
  soft_probe_prompt_cap: 5  # 5 different prompts
  generations: 4            # 4 responses per prompt
```

### Result
- **5 different prompts** × **4 generations** = **20 API calls per probe**

### Visual Breakdown
```
Probe: HijackHateHumans

WITH soft_probe_prompt_cap: 5, generations: 4:
├─ Prompt 1 → [Response 1, Response 2, Response 3, Response 4]  (4 API calls)
├─ Prompt 2 → [Response 1, Response 2, Response 3, Response 4]  (4 API calls)
├─ Prompt 3 → [Response 1, Response 2, Response 3, Response 4]  (4 API calls)
├─ Prompt 4 → [Response 1, Response 2, Response 3, Response 4]  (4 API calls)
└─ Prompt 5 → [Response 1, Response 2, Response 3, Response 4]  (4 API calls)
Total: 20 API calls
```

---

## Comparison Table

| Parameter | Controls | Purpose | Example | Enforced? |
|-----------|----------|---------|---------|-----------|
| `soft_probe_prompt_cap` | Number of **different prompts** | Limits test scope | `5` = test 5 different attack patterns | **No** (soft/advisory) |
| `generations` | Times to run **each prompt** | Statistical confidence | `4` = get 4 responses per prompt | **Yes** (always enforced) |

---

## Formula

**Total API Calls = `soft_probe_prompt_cap` × `generations`**

### Examples

| `soft_probe_prompt_cap` | `generations` | Total API Calls per Probe |
|-------------------------|---------------|---------------------------|
| 5 | 1 | 5 |
| 5 | 4 | 20 |
| 10 | 2 | 20 |
| 50 | 1 | 50 |
| 50 | 4 | 200 |

---

## Important Considerations

### `soft_probe_prompt_cap` Limitations

1. **Not a hard limit**: The "soft" in the name means it's advisory
2. **Probe minimums**: Some probes have minimum requirements that override the cap
3. **Probe design**: Some probes may ignore the cap entirely
4. **Default value**: If not specified, defaults to 256

### Why Probes May Exceed the Cap

- **Probe minimums**: Some probes require a minimum number of attempts (e.g., 51 for `ansiescape.AnsiEscaped`)
- **Probe design**: Probes may be designed to ignore the soft cap
- **Default behavior**: If not set, Garak uses a default value (typically 256)

### Alternative Ways to Limit Attempts

1. **`max_probes`**: Limits the number of probe classes (not attempts per probe)
   ```yaml
   run:
     max_probes: 3  # Limits to 3 probe classes total
   ```

2. **Specify individual probes**: Instead of using module names, specify exact probe classes
   ```bash
   --probes promptinject.HijackHateHumans promptinject.SomeOtherProbe
   ```

3. **`timeout`**: Indirectly limits attempts by stopping after a time limit
   ```yaml
   run:
     timeout: 600  # Stop after 600 seconds
   ```

---

## Configuration Examples

### Speed-Optimized (Fast PR Gate)
```yaml
run:
  soft_probe_prompt_cap: 5   # Few prompts per probe
  generations: 1              # Single generation per prompt
  max_probes: 3                # Limit probe classes
```
**Result**: ~5 API calls per probe, very fast execution

### Balanced (Good Coverage)
```yaml
run:
  soft_probe_prompt_cap: 20   # Moderate number of prompts
  generations: 2               # 2 generations for some confidence
  max_probes: 10               # More probe classes
```
**Result**: ~40 API calls per probe, moderate execution time

### Thorough (Comprehensive Testing)
```yaml
run:
  soft_probe_prompt_cap: 50   # Many prompts per probe
  generations: 4               # 4 generations for high confidence
  # max_probes: (not set, run all probes)
```
**Result**: ~200 API calls per probe, thorough but slow execution

---

## Real-World Example

### Scenario: Testing `promptinject.HijackHateHumans`

**Configuration:**
```yaml
run:
  soft_probe_prompt_cap: 5
  generations: 1
```

**Expected behavior:**
- Probe tries to generate 5 different prompt injection attempts
- Each attempt is sent to the model once
- Total: 5 API calls

**Actual behavior (if probe ignores cap):**
- Probe may generate 256 attempts (its default)
- Each attempt sent once
- Total: 256 API calls

**Why?** The probe may have a minimum requirement or ignore the soft cap.

---

## Best Practices

1. **For PR gates**: Use low values for speed
   - `soft_probe_prompt_cap: 5-10`
   - `generations: 1`
   - `max_probes: 3-5`

2. **For comprehensive testing**: Use higher values
   - `soft_probe_prompt_cap: 50+`
   - `generations: 4`
   - Don't set `max_probes` (run all probes)

3. **For statistical confidence**: Increase `generations`
   - `generations: 4-8` for better confidence
   - Keep `soft_probe_prompt_cap` moderate to balance time

4. **Monitor actual behavior**: Check reports to see if probes respect the cap
   - Look at `run.soft_probe_prompt_cap` in the report JSONL
   - Count actual attempts in the report
   - Adjust if probes ignore the cap

---

## Related Configuration Parameters

- **`max_probes`**: Limits the number of probe classes (not attempts per probe)
- **`timeout`**: Maximum execution time in seconds
- **`parallel_attempts`**: Number of attempts to run in parallel
- **`temperature`**: Model temperature (affects response variability)

---

## References

- [Garak Documentation](https://garak.ai/)
- Garak version: 0.13.3
- Config file: `garak_PR_config.yaml`

---

## Summary

- **`generations`**: How many times to run the **same prompt** → Always enforced
- **`soft_probe_prompt_cap`**: How many **different prompts** per probe → Advisory only, may be ignored
- **Total API calls** = `soft_probe_prompt_cap` × `generations` (if cap is respected)
- Use `max_probes` and `timeout` for additional control over test scope and duration

