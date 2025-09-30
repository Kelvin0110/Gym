import json

import pandas as pd
from datasets import Dataset, load_dataset


# Helper function to parse the JSON and extract the user content
def _extract_user_content(json_string: str) -> str | None:
    """
    Parses a JSON string from the 'responses_create_params' column,
    finds the message where role is 'user', and returns its content.
    """
    if not isinstance(json_string, str):
        return None
    try:
        data = json.loads(json_string)
        input_list = data.get("input", [])
        if not isinstance(input_list, list):
            return None

        for message in input_list:
            if isinstance(message, dict) and message.get("role") == "user":
                return message.get("content")
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def _extract_user_from_prompt_message(json_string: str) -> str | None:
    """
    Parses a JSON string from the 'prompt_message' column,
    and returns its content.
    """
    if isinstance(json_string, dict):
        return json_string.get("content", "")
    return None


def combine_and_upload(dataset1: Dataset, dataset2: Dataset, new_repo_name: str, hf_username: str):
    """
    Combines two datasets based on a nested JSON value and uploads the
    result as a new private dataset.
    """
    # 1. Convert datasets to pandas DataFrames
    print("🔄 Converting datasets to pandas DataFrames...")
    df1 = dataset1.to_pandas()
    df2 = dataset2.to_pandas()

    # 2. Extract the join key from the JSON column in both DataFrames
    print("🔎 Extracting user content from JSON to create join key...")
    df2["join_key"] = df2["responses_create_params"].apply(_extract_user_content)
    df1["join_key"] = df1["prompt_message"].apply(_extract_user_from_prompt_message)

    # Drop rows where the key could not be extracted to ensure a clean merge
    df1.dropna(subset=["join_key"], inplace=True)
    df2.dropna(subset=["join_key"], inplace=True)

    print(f"Found {len(df1)} valid join keys in the first dataset.")
    print(f"Found {len(df2)} valid join keys in the second dataset.")

    df1_rename_map = {
        "reward": "verl_reward",
    }
    df2_rename_map = {
        "reward": "nemogym_reward",
    }

    df1.rename(columns=df1_rename_map, inplace=True)
    df2.rename(columns=df2_rename_map, inplace=True)

    # 3. Perform an inner merge using the different column names
    print("🤝 Merging datasets on the extracted user content...")
    combined_df = pd.merge(
        df1,
        df2,
        on="join_key",  # Simplified since left_on and right_on are the same
        how="inner",
    )

    # 4. Clean up the DataFrame by dropping the temporary join key
    combined_df = combined_df.drop(columns=["join_key"])

    # 5. Convert back to a Hugging Face Dataset and upload
    if len(combined_df) == 0:
        print("\n⚠️ Merge resulted in an empty dataset. No matching rows were found. Aborting upload.")
        return

    combined_dataset = Dataset.from_pandas(combined_df)
    print("\n✅ Merge complete. Resulting dataset preview:")
    print(combined_dataset)

    repo_id = f"{hf_username}/{new_repo_name}"
    print(f"\n🚀 Uploading dataset to '{repo_id}' as a private repository...")

    repo_url = combined_dataset.push_to_hub(repo_id=repo_id, private=True)
    print(f"\n🎉 Successfully uploaded! Find your private dataset here: {repo_url}")


# --- Configuration and Example Usage ---


# Since the datasets are specific, we'll create dummy ones for a runnable example.
# In your actual use case, you would load them from the Hub like this:
dataset1 = load_dataset("Nexusflow/abhibha-workbench-rollouts-verl-grp-size-1-fair-comparison", split="train")
dataset2 = load_dataset("Nexusflow/abhibha-gpt-rollouts-fixed-grp-size-1-completions", split="train")

combine_and_upload(
    dataset1=dataset1,
    dataset2=dataset2,
    new_repo_name="gpt-workbench-diffs-verl-vs-workbench-grp-size-1-fair-comparison",
    hf_username="Nexusflow",
)

# Nexusflow/gpt-workbench-diffs-verl-vs-workbench-grp-size-1-fair-comparison
