from datasets import load_dataset


# --- 1. Load your Hugging Face Dataset ---
# Replace "go_emotions" with your actual dataset name (e.g., "username/my-dataset-name").
# I'm using the 'go_emotions' dataset here as a public example.
print("Loading dataset...")
ds = load_dataset("Nexusflow/250828-gpt-workbench-grp-size-16-verl-train-rollouts", split="train")


# --- 2. Calculate the totals ---
total_ones = 0
total_length = 0

# Iterate through each row in the dataset
print("Calculating totals...")
for row in ds:
    # Access the "rewards" column for each row
    reward_list = row["rewards"]

    # sum() on a list of 1s and 0s is a fast way to count the 1s
    total_ones += sum(reward_list)

    # Get the length of the list in the current row
    total_length += len(reward_list)

# --- 3. Calculate the final ratio ---
if total_length > 0:
    ratio = total_ones / total_length
else:
    ratio = 0

# --- 4. Print the results ---
print("\n--- Calculation Complete ---")
print(f"Total number of 1s: {total_ones}")
print(f"Total length of all lists: {total_length}")
print(f"Ratio of ones to total length: {ratio:.4f}")
