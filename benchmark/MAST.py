# %%
from huggingface_hub import hf_hub_download
import pandas as pd
import json

REPO_ID = "mcemri/MAD"
FILENAME = "MAD_full_dataset.json"

file_path =  hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
with open(file_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records (full dataset).")

# %%
print(json.dumps(data[0], indent=4))

# %%
import json

failed_count = 0
for item in data:
    annotations = item.get("mast_annotation", {})
    if any(v != 0 for v in annotations.values()):
        failed_count += 1

print(f"Failed datapoints: {failed_count}")
print(f"Total datapoints: {len(data)}")

# %%
import pickle

# Extract all trajectories from the main dataset
trajectories = []
for item in data:
    if "trace" in item and "trajectory" in item["trace"]:
        trajectories.append(item["trace"]["trajectory"])

# Save trajectories to a pickle file
with open("MAD_full_trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)

print(f"Saved {len(trajectories)} trajectories to MAD_full_trajectories.pkl")

# %%
# Load trajectories from the pickle file
with open("MAD_full_trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)

print(f"Loaded {len(trajectories)} trajectories from MAD_full_trajectories.pkl")

print(trajectories[0])


# %%
FILENAME = "MAD_human_labelled_dataset.json"

file_path =  hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
with open(file_path, "r") as f:
    data_human = json.load(f)

print(f"Loaded {len(data_human)} records (human labelled).")

# %%
print(json.dumps(data_human[0], indent=4))

# %%
failed_count_human = 0

for item in data_human:
    annotations_list = item.get("annotations", [])
    for fm in annotations_list:
        if (
            fm.get("annotator_1", False)
            or fm.get("annotator_2", False)
            or fm.get("annotator_3", False)
        ):
            failed_count_human += 1
            break


print(f"Failed datapoints: {failed_count_human}")
print(f"Total datapoints: {len(data_human)}")

# %%

print(f"Failed datapoints: {failed_count_human + failed_count}")
print(f"Total datapoints: {len(data_human) + len(data)}")

# %%
import matplotlib.pyplot as plt

# Number of failed and successful datapoints
# total_failed = failed_count_human + failed_count
# total = len(data_human) + len(data)
total_failed = failed_count
total = len(data)
total_success = total - total_failed

# Data for bars
categories = ['Failed', 'Success']
counts = [total_failed, total_success]
percentages = [c / total * 100 for c in counts]

# Create figure
plt.figure(figsize=(6, 4))
bars = plt.bar(categories, counts, color=['#e76f51', '#2a9d8f'])  # coral and teal colors for better contrast
plt.title('Datapoint Statistics')
plt.ylabel('Count')

# Make y scale bigger by setting a higher limit
max_count = max(counts)
plt.ylim(0, max_count * 1.2)

# Annotate bars with count and percentage
for bar, count, percent in zip(bars, counts, percentages):
    height = bar.get_height()
    plt.annotate(f'{count} ({percent:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.show()

# %%
trajectory = data[0]["trace"].get("trajectory", None)
import re

if trajectory is not None and isinstance(trajectory, str):
    # Regex for log prefix (e.g., [2025-31-03 19:09:41 INFO])
    log_prefix_re = re.compile(r'(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Z]+\])')
    # Insert newline before every log prefix (except if it's at the very start)
    trajectory = log_prefix_re.sub(r'RUI: \1', trajectory).lstrip()
    # Remove lines that match "flask app.py did not start for online log" match it and substitute it with empty string

if trajectory is not None:
    with open("trajectory_data0.json", "w") as f:
        import json
        json.dump(trajectory, f, indent=2)
    print("Trajectory for data[0] written to trajectory_data0.json")
else:
    print("No trajectory found in data[0]")


# %%
trajectory1 = data_human[0]["trace"]

with open("trajectory_data0_human.json", "w") as f:
    import json
    json.dump(trajectory1, f, indent=2)
print("Trajectory for data[0] written to trajectory_data0_human.json")
# %%
