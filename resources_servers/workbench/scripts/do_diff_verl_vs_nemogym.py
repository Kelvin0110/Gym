import pandas as pd
from datasets import load_dataset


# Login using e.g. `huggingface-cli login` to access this dataset
ds_nemogym = load_dataset("Nexusflow/abhibha-gpt-rollouts-fixed-grp-size-16-completions", split="train")


ds_verl = load_dataset("Nexusflow/250828-gpt-workbench-grp-size-16-verl-train-rollouts", split="train")


import json
from collections import defaultdict

import numpy as np
from datasets import Dataset


original_ds = ds_nemogym

# --- 2. Extract user queries and group data ---
print("Extracting user queries and grouping the data...")
grouped_data = defaultdict(list)


def get_user_query(example):
    """Extracts the user message content from the response_create_params column."""
    try:
        # Load the JSON string from the column
        params = json.loads(example["responses_create_params"])
        # Find the message with the 'user' role
        for message in params.get("input", []):
            if message.get("role") == "user":
                return message.get("content")
    except (json.JSONDecodeError, TypeError):
        # Handle cases where the data is not a valid JSON string or is None
        return None
    return None


# Iterate through the dataset to group all original data by the user query
for row in original_ds:
    query = get_user_query(row)
    if query:
        grouped_data[query].append(row)

print(f"Found {len(grouped_data)} unique user queries.")

# --- 3. Create the new dataset structure ---
print("Processing groups and generating rewards...")
new_dataset_rows = []
for query, rows in grouped_data.items():
    # For each unique query, generate rewards
    # Here, we generate random rewards for demonstration purposes

    rewards = [row["reward"] for row in rows]
    mean_reward = np.mean(rewards)
    # You can decide how to aggregate other columns.
    # For this example, we'll just store the count of occurrences and the list of original IDs.
    # original_ids = [row['id'] for row in rows]

    new_dataset_rows.append(
        {
            "user_query": query,
            "occurrences": len(rows),
            # 'original_ids': original_ids,
            "reward": rewards,
            "mean_reward": mean_reward,
        }
    )

# --- 4. Convert the processed list to a Hugging Face Dataset ---
print("Creating the new Hugging Face dataset...")
final_ds = Dataset.from_list(new_dataset_rows)

print("\n--- Sample of the new dataset ---")
print(final_ds[0])
print("---------------------------------")

# --- 5. Upload the new dataset to the Hugging Face Hub ---
# Make sure you have authenticated with 'huggingface-cli login'
# Replace 'YOUR_HF_USERNAME' with your actual username
dataset_name = "Nexusflow/abhibha-gpt-rollouts-fixed-grp-size-16-completions-consolidated"
repo_id = f"{dataset_name}"

print(f"Uploading the dataset to '{repo_id}' as a private dataset...")
final_ds.push_to_hub(repo_id, private=True)

print("\n🚀 All done! Your new dataset has been successfully uploaded.")


# --- 2. Align (Join) Datasets using Pandas ---
print("\nConverting to pandas and aligning datasets...")

# Convert datasets to pandas DataFrames
df_final = final_ds.to_pandas()
df_verl = ds_verl.to_pandas()

# Perform an inner join
# 'left_on' is the key from the first DataFrame, 'right_on' is from the second
# Pandas automatically adds suffixes (_x, _y) to columns with the same name
aligned_df = pd.merge(df_final, df_verl, left_on="user_query", right_on="problem")

aligned_df = aligned_df.rename(columns={"mean_reward_x": "mean_reward_nemogym", "mean_reward_y": "mean_reward_verl"})


# Convert the merged DataFrame back to a Hugging Face Dataset
aligned_ds = Dataset.from_pandas(aligned_df)

print(f"Alignment complete. The new dataset has {len(aligned_ds)} rows.")
print("\n--- Sample of the aligned dataset ---")
print(aligned_ds[0])
print("-----------------------------------")


# --- 3. Upload the Aligned Dataset ---
aligned_repo_id = "Nexusflow/gpt-workbench-grp-size-16-verl-vs-nemogym-rollouts"
print(f"\nUploading the full aligned dataset to '{aligned_repo_id}'...")

aligned_ds.push_to_hub(aligned_repo_id, private=True)

print("✅ First upload complete.")


# --- 4. Filter for Rows with Different Mean Rewards ---
print("\nFiltering for rows with different mean rewards...")

# Identify the mean_reward columns. They will be named 'mean_reward_x' and 'mean_reward_y'.
# '_x' comes from the left DataFrame (final_ds) and '_y' from the right (ds_verl)
filtered_df = aligned_df[aligned_df["mean_reward_nemogym"] != aligned_df["mean_reward_verl"]]

# Convert the filtered DataFrame back to a Hugging Face Dataset
filtered_ds = Dataset.from_pandas(filtered_df)

print(f"Found {len(filtered_ds)} rows where mean rewards are different.")


# --- 5. Upload the Filtered Dataset ---
filtered_repo_id = "Nexusflow/gpt-workbench-grp-size-16-verl-vs-nemogym-diffs"
print(f"\nUploading the filtered dataset to '{filtered_repo_id}'...")

filtered_ds.push_to_hub(filtered_repo_id, private=True)

print("✅ Second upload complete.")
print("\n🚀 All tasks finished successfully!")
