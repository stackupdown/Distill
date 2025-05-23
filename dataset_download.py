from datasets import load_dataset
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path of the whole dataset")
    parser.add_argument("--top_k", type=int, default=1000, help="top k samples")
    parser.add_argument("--output_dataset_path", required=True) # default = dataset_path.split('/')[-1]
    return parser.parse_args()

args = parse_args()

# 加载原始数据集
all_set_path = args.dataset_path
dataset = load_dataset(all_set_path, split="train")

if args.top_k:
    subset = dataset.select(range(args.top_k))
else:
    subset = dataset

subset_path = args.output_dataset_path
subset_dir = os.path.dirname(subset_path)

if not os.path.exists(subset_dir):
    os.makedirs(subset_dir, exist_ok=True)
    print("create dir {}".format(subset_dir))

subset.to_json(subset_path)

# 加载新数据集, train是默认的split
loaded_dataset = load_dataset("json", data_files=subset_path, split="train")
print(loaded_dataset[0])
