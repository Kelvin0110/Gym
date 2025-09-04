import json
import random
import os
def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def make_examples(file_path):
    data = read_jsonl(file_path)
    #randomly select 5 examples
    examples = random.sample(data, 5)
    return examples

def write_jsonl(examples, input_file_path):
    parent_folder = os.path.dirname(input_file_path)
    output_file_path = os.path.join(parent_folder, "examples.jsonl")
    with open(output_file_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    file_paths = [
        "/lustre/fsw/portfolios/llmservice/users/rgala/repos/nemo-gym/resources_servers/offline_search/data/MCQA_syn_gpqa_1_2_difficulty_filtered/train.jsonl",
        "/lustre/fsw/portfolios/llmservice/users/rgala/repos/nemo-gym/resources_servers/offline_search/data/MCQA_syn_hle/train.jsonl"
    ]
    for file_path in file_paths:
        examples = make_examples(file_path)
        write_jsonl(examples, file_path)