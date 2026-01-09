import json
from pathlib import Path

report_path = Path(r"C:\Users\Virochan\.local\share\garak\garak_runs\garak.83d49cf1-0efd-4792-9381-1a95639859b8.report.jsonl")

print(f"Reading: {report_path}")
print("=" * 80)

eval_count = 0
attempt_count = 0
other_count = 0

with open(report_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            row = json.loads(line)
            entry_type = row.get("entry_type")
            
            if entry_type == "eval":
                eval_count += 1
                if eval_count <= 3:  # Show first 3 eval entries
                    print(f"\n--- Eval Entry #{eval_count} (Line {line_num}) ---")
                    print(json.dumps(row, indent=2))
                    print()
            
            elif entry_type == "attempt":
                attempt_count += 1
                if attempt_count <= 5:  # Show first 5 attempt entries
                    print(f"\n--- Attempt Entry #{attempt_count} (Line {line_num}) ---")
                    print(json.dumps(row, indent=2))
                    print()
            
            else:
                other_count += 1
                if other_count <= 1:  # Show first other entry
                    print(f"\n--- Other Entry Type: {entry_type} (Line {line_num}) ---")
                    print(json.dumps(row, indent=2))
                    print()
                    
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}")

print("=" * 80)
print(f"Summary:")
print(f"  Total eval entries: {eval_count}")
print(f"  Total attempt entries: {attempt_count}")
print(f"  Other entries: {other_count}")

