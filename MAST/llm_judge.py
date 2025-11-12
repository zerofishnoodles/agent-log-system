# %%
"""
Visualization and plotting utilities for LLM evaluation results.
This module focuses on plotting and analyzing evaluation results in Jupyter notebook style.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from evaluation import LLMEvaluator

from huggingface_hub import hf_hub_download

import pandas as pd
import json
import asyncio

#%% 
REPO_ID = "mcemri/MAD"
FILENAME = "MAD_full_dataset.json"

file_path =  hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
with open(file_path, "r") as f:
    data = json.load(f)
ground_truth = [0 if any(val == 1 for val in record["mast_annotation"].values()) else 1 for record in data]
print(f"Loaded {len(data)} records (full dataset).")



# %%

async def run_evaluation_example():
    """Example of how to run evaluation with the new structure."""
    # Load traces
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import json

    REPO_ID = "mcemri/MAD"
    FILENAME = "MAD_full_dataset.json"

    file_path =  hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records (full dataset).")

    full_trace_list = data
    # Initialize evaluator
    evaluator = LLMEvaluator(
        model="openai/gpt-oss-120b",
        base_url="http://localhost:30080/v1",
        api_key="KEY",
        max_model_length=73719,
        routing_logic="roundrobin",
    )
    
    # Run evaluation (this will use semantic caching)
    times_info_list, results = await evaluator.evaluate_traces(full_trace_list, cache_dir='saved_results', cache_enabled=False)
    
    # Save results
    model_name = "openai/gpt-oss-120b"
    model_clean = model_name.replace("/", "_")
    with open(f'saved_results/{model_clean}_73719_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(f'saved_results/{model_clean}_73719_times_info.pkl', 'wb') as f:
        pickle.dump(times_info_list, f)
    
    return times_info_list, results

times_info_list, results = await run_evaluation_example()
# %%
# Load evaluation results
model = "openai/gpt-oss-20b"
dirname = 'saved_results'
with open(f'{dirname}/{model.replace("/", "_")}_73719_times_info.pkl', 'rb') as f:
    times_info_list = pickle.load(f)
with open(f'{dirname}/{model.replace("/", "_")}_73719_results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"Total evaluations: {len(times_info_list)}")
times_info_list_without_errors = [time for time in times_info_list if time["error"] is None]
print(f"Total evaluations without errors: {len(times_info_list_without_errors)}")
print(f"Average time per evaluation without errors: {sum([time['duration_seconds'] for time in times_info_list_without_errors]) / len(times_info_list_without_errors)} seconds")


# %%
dirname = 'saved_results'
# Compare two models - load data for comparison
with open(f'{dirname}/openai_gpt-oss-20b_73719_times_info.pkl', 'rb') as f:
    times_info_list_oss_20b = pickle.load(f)
with open(f'{dirname}/openai_gpt-oss-120b_73719_times_info.pkl', 'rb') as f:
    times_info_list_oss_120b = pickle.load(f)

durations_oss_20b = [time['duration_seconds'] for time in times_info_list_oss_20b if time["duration_seconds"]]
durations_oss_120b = [time['duration_seconds'] for time in times_info_list_oss_120b if time["duration_seconds"]]

# Show mean duration for each model
mean_duration_oss_20b = np.mean(durations_oss_20b)
mean_duration_oss_120b = np.mean(durations_oss_120b)
print(f"Average evaluation duration for oss-20b: {mean_duration_oss_20b:.2f} seconds")
print(f"Average evaluation duration for oss-120b: {mean_duration_oss_120b:.2f} seconds")


plt.figure(figsize=(9, 6))

# Determine common bin edges for both histograms
all_durations = np.hstack((durations_oss_20b, durations_oss_120b))
bin_edges = np.histogram_bin_edges(all_durations, bins='auto')

bar_width = (bin_edges[1] - bin_edges[0]) * 0.4  # To separate bars

# Compute histograms
hist_20b, _ = np.histogram(durations_oss_20b, bins=bin_edges)
hist_120b, _ = np.histogram(durations_oss_120b, bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot as side-by-side bars
plt.bar(
    bin_centers - bar_width/2, hist_20b, width=bar_width, 
    color='#43aa8b', label='oss-20b', align='center', edgecolor='black'
)
plt.bar(
    bin_centers + bar_width/2, hist_120b, width=bar_width, 
    color='#f3722c', label='oss-120b', align='center', edgecolor='black'
)

plt.title('Histogram of Evaluation Durations: oss-20b vs. oss-120b')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 114688
# %%
# Load results for success/failure analysis
with open(f'{dirname}/gpt-oss-20b_results_checkpoint.pkl', 'rb') as f:
    results_oss_20b = pickle.load(f)
with open(f'{dirname}/gpt-oss-120b_results_checkpoint.pkl', 'rb') as f:
    results_oss_120b = pickle.load(f)

# %%
def extract_success_fail_counts(results, ground_truth):
    failed = 0
    success = 0
    for i, r in enumerate(results):
        # Find a substring like 'B. yes' or 'B. no', case-insensitive for yes/no, B. is always capital.
        r_str = str(r)
        idx = r_str.find("B.")
        if idx != -1:
            after_b = r_str[idx+3:].lstrip()
            if (after_b.lower().startswith("yes") and ground_truth[i] == 1) or (after_b.lower().startswith("no") and ground_truth[i] == 0):
                success += 1
            else:
                failed += 1

    return failed, success

# %%

with open(f'{dirname}/gpt-oss-20b_times_info_checkpoint.pkl', 'rb') as f:
    times_info_list_oss_20b = pickle.load(f)
with open(f'{dirname}/gpt-oss-120b_times_info_checkpoint.pkl', 'rb') as f:
    times_info_list_oss_120b = pickle.load(f)

durations_oss_20b = [time['duration_seconds'] for time in times_info_list_oss_20b if time["error"] is None]
durations_oss_120b = [time['duration_seconds'] for time in times_info_list_oss_120b if time["error"] is None]

print(f"Total evaluations without errors (oss-20b): {len(times_info_list_oss_20b)}")
print(f"Total evaluations without errors (oss-120b): {len(times_info_list_oss_120b)}")
# %%

# Error rate comparison plot for gpt-oss-20b across model lengths

model_name = "openai_gpt-oss-120b"
dirname = "saved_results"
model_lens = [73719, 81902, 98304, 114688, 131072]

accuracies = []
for model_len in model_lens:
    times_info_path = f"{dirname}/{model_name}_{model_len}_times_info.pkl"
    results_path = f"{dirname}/{model_name}_{model_len}_results.pkl"
    with open(times_info_path, "rb") as f:
        times_info = pickle.load(f)
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    fail_count, success_count = extract_success_fail_counts(results, ground_truth)
    accuracy = success_count / (success_count + fail_count)
    accuracies.append(accuracy)
# model_lens.append("o1-paper")


# Plot error rates (as a barplot for easier comparison)
plt.figure(figsize=(8, 6))
bars = plt.bar([str(l) for l in model_lens], accuracies, color='#FF7043', alpha=0.88)
plt.title("gpt-oss-120b Accuracy vs Model Length", fontsize=16)
plt.xlabel("Model Length", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.25)

for i, bar in enumerate(bars):
    accuracy = accuracies[i]
    if not np.isnan(accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, accuracy + .025, f"{accuracy:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()

# %%

# Plot the average duration for different model lengths and two models: gpt-oss-20b and gpt-oss-120b

model_names = ["openai_gpt-oss-20b", "openai_gpt-oss-120b"]
model_lens = [73719, 81902, 98304, 114688, 131072]
dirname = "saved_results"
colors = ["#1E88E5", "#FF7043"]
avg_durations = {}

for model_name in model_names:
    avg_durations[model_name] = []
    for model_len in model_lens:
        times_info_path = f"{dirname}/{model_name}_{model_len}_times_info.pkl"
        try:
            with open(times_info_path, "rb") as f:
                times_info = pickle.load(f)
            durations = [item["duration_seconds"] for item in times_info if item.get("error") is None]
            if len(durations) > 0:
                avg_duration = np.mean(durations)
            else:
                avg_duration = float('nan')
        except Exception as e:
            avg_duration = float('nan')
        avg_durations[model_name].append(avg_duration)

plt.figure(figsize=(8, 6))
for idx, model_name in enumerate(model_names):
    plt.plot(
        [str(l) for l in model_lens],
        avg_durations[model_name],
        marker="o",
        color=colors[idx],
        label=model_name.replace("openai_", "")
    )

plt.title("Average Evaluation Duration vs Context Length", fontsize=16)
plt.xlabel("Context Length", fontsize=13)
plt.ylabel("Average Duration (seconds)", fontsize=13)
plt.ylim(bottom=0)
plt.grid(axis="y", linestyle="--", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# %%

# Plot a figure to compare "roundrobin" (multi-model) and "same model" strategy for model_len = 81902

model_len = 81902
dirname = "saved_results"

# These are example naming conventions; adjust if needed
# Assume two models: openai_gpt-oss-20b and openai_gpt-oss-120b
model_names = ["openai_gpt-oss-20b", "openai_gpt-oss-120b"]

# Single model strategy (use one model for the whole evaluation)
avg_durations_single = []
for model_name in model_names:
    times_info_path = f"{dirname}/{model_name}_{model_len}_times_info.pkl"
    try:
        with open(times_info_path, "rb") as f:
            times_info = pickle.load(f)
        durations = [item["duration_seconds"] for item in times_info if item.get("error") is None]
        if durations:
            avg_durations_single.append(np.mean(durations))
        else:
            avg_durations_single.append(float("nan"))
    except Exception as e:
        avg_durations_single.append(float("nan"))

# Roundrobin strategy (use both models with roundrobin routing)
# Assume the roundrobin results are saved as a special file, e.g., "openai_gpt-oss-20b+openai_gpt-oss-120b_81902_times_info.pkl"
times_info_path_rr = f"{dirname}/openai_gpt-oss-120b_{model_len}_roundrobin_times_info.pkl"
try:
    with open(times_info_path_rr, "rb") as f:
        times_info_rr = pickle.load(f)
    durations_rr = [item["duration_seconds"] for item in times_info_rr if item.get("error") is None]
    avg_duration_rr = np.mean(durations_rr) if durations_rr else float("nan")
except Exception as e:
    avg_duration_rr = float("nan")

# Prepare data for plotting
labels = [
    "gpt-oss-20b (single)", 
    "gpt-oss-120b (single)", 
    "RoundRobin (20b+120b)"
]
avg_durations = avg_durations_single + [avg_duration_rr]
bar_colors = ["#1E88E5", "#FF7043", "#43A047"]

plt.figure(figsize=(7,5))
bars = plt.bar(labels, avg_durations, color=bar_colors)
plt.ylabel("Average Duration (seconds)", fontsize=13)
plt.title("Average Evaluation Duration (Context Length 81902)\nSingle Model vs Roundrobin", fontsize=15)
plt.ylim(bottom=0)

# Annotate bar values
for bar in bars:
    height = bar.get_height()
    plt.annotate(
        f'{height:.1f}',
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.tight_layout()
plt.show()


# %%

# Compare accuracy for roundrobin and others at model length 81902

model_len = 81902
dirname = "saved_results"
ground_truth = ground_truth  # already loaded at top

model_names = [
    ("openai_gpt-oss-20b", "gpt-oss-20b"),
    ("openai_gpt-oss-120b", "gpt-oss-120b"),
]

accuracies = []
labels = []
for model_file_name, model_label in model_names:
    results_path = f"{dirname}/{model_file_name}_{model_len}_results.pkl"
    try:
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        fail_count, success_count = extract_success_fail_counts(results, ground_truth)
        accuracy = success_count / (success_count + fail_count) if (success_count + fail_count) > 0 else float("nan")
    except Exception as e:
        accuracy = float("nan")
    accuracies.append(accuracy)
    labels.append(model_label)

# Roundrobin: assumes results file is "openai_gpt-oss-120b_81902_roundrobin_results.pkl"
model_label_rr = "RoundRobin (20b+120b)"
results_path_rr = f"{dirname}/openai_gpt-oss-120b_{model_len}_roundrobin_results.pkl"
try:
    with open(results_path_rr, "rb") as f:
        results_rr = pickle.load(f)
    fail_count_rr, success_count_rr = extract_success_fail_counts(results_rr, ground_truth)
    accuracy_rr = success_count_rr / (success_count_rr + fail_count_rr) if (success_count_rr + fail_count_rr) > 0 else float("nan")
except Exception as e:
    accuracy_rr = float("nan")
accuracies.append(accuracy_rr)
labels.append(model_label_rr)

colors = ["#1E88E5", "#FF7043", "#43A047"]

plt.figure(figsize=(7,5))
bars = plt.bar(labels, accuracies, color=colors, alpha=0.93)
plt.ylabel("Accuracy", fontsize=13)
plt.title("Accuracy Comparison (Context Length 81902)\nSingle Model vs Roundrobin", fontsize=15)
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.22)

for i, bar in enumerate(bars):
    accuracy = accuracies[i]
    if not np.isnan(accuracy):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.025,
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

plt.tight_layout()
plt.show()

# %%
