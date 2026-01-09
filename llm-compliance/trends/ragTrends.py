import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load the data
# Get the script's directory and resolve the path relative to it
script_dir = Path(__file__).parent
csv_path = script_dir.parent / 'reports' / 'metrics_history.csv'
df = pd.read_csv(csv_path)

# 2. Convert timestamp to datetime objects for proper scaling
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Sort by timestamp to ensure the lines connect correctly
df = df.sort_values('timestamp')

# 4. Set the visual style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# 5. Plot the key RAGAS metrics
# Note: Column names in CSV have '_score' suffix
metrics_to_plot = ['answer_correctness_score', 'context_precision_score', 'context_recall_score', 'faithfulness_score']

for metric in metrics_to_plot:
    if metric in df.columns:
        sns.lineplot(data=df, x='timestamp', y=metric, marker='o', label=metric)

# 6. Formatting the chart
plt.title('RAG Evaluation Metrics Over Time', fontsize=16)
plt.xlabel('Run Timestamp', fontsize=12)
plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
plt.xticks(rotation=45)
plt.ylim(-0.05, 1.05) # Metrics are usually between 0 and 1
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 7. Show or Save
# Ensure charts directory exists
charts_dir = script_dir / 'charts'
charts_dir.mkdir(exist_ok=True)

# Save the figure
output_path = charts_dir / 'rag_performance_trend.png'
plt.savefig(output_path)
print(f"Chart saved to: {output_path}")