from datasets import load_dataset


# --- Configuration ---
# 1. Replace this with the name of your Hugging Face dataset
DATASET_NAME = "Nexusflow/250828-gpt-workbench-grp-size-16-verl-train-rollouts"
# 2. This is the string you want to find
SEARCH_STRING = "Error executing tool"

# --- NEW: Configuration for uploading ---
# 3. Replace this with your Hugging Face username
HF_USERNAME = "YOUR_HF_USERNAME"
# 4. Choose a name for your new private dataset
NEW_DATASET_NAME = "filtered-rollouts-with-errors"


try:
    # --- 1. Load the Dataset ---
    print(f"Loading dataset: '{DATASET_NAME}'...")
    ds = load_dataset(DATASET_NAME, split="train")
    print("Dataset loaded successfully.")

    # --- 2. Define the Filtering Function ---
    def contains_error_string(example):
        """Checks if the 'rollouts' column contains the search string."""
        rollout_text = example.get("rollouts")
        return SEARCH_STRING in str(rollout_text)

    # --- 3. Apply the Filter ---
    print(f"Filtering for rows where 'rollouts' contains '{SEARCH_STRING}'...")
    filtered_ds = ds.filter(contains_error_string)

    # --- 4. Display the Results and Upload ---
    print("\n--- Results ---")
    num_found = len(filtered_ds)

    if num_found > 0:
        print(f"✅ Found {num_found} rows containing the error string.")

        # --- NEW: Upload the filtered dataset ---
        repo_id = "Nexusflow/abhibha-gpt-verl-rollouts-investigate-error"
        print(f"\nUploading the {num_found} filtered rows to a new private dataset: '{repo_id}'")

        filtered_ds.push_to_hub(
            repo_id=repo_id,
            private=True,  # This makes the new dataset private
        )

        print("✅ Upload complete!")
        # ----------------------------------------

        # You can still print samples if you want
        print("\nHere are the first few examples from the uploaded dataset:")
        for i in range(min(3, num_found)):
            print(f"\n--- Example {i + 1} ---")
            print(filtered_ds[i]["rollouts"])
            print("-----------------")
    else:
        print("❌ No rows were found containing the specified error string. Nothing to upload.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check if the dataset name is correct and if it contains a 'rollout' column.")
