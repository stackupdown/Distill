from datasets import load_dataset
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="path of the whole dataset")
    parser.add_argument("--small_dataset_file", type=str, help="path of the small dataset")
    parser.add_argument("--top_k", type=int, default=1000, help="top k samples")
    return parser.parse_args()

args = parse_args()

# 加载原始数据集
# all_set_path = "../QwQ-LongCoT-130K_mmap"
all_set_path = args.dataset_path
dataset = load_dataset(all_set_path, split="train")

subset = dataset.train_test_split(test_size=0.1, seed=42)['test']
subset.to_json(subset_path)

# subset = dataset

if args.top_k:
    subset = dataset.select(range(args.top_k))
# print("subset", subset)
# JSON 文件
# subset_path = "../QwQ-LongCoT-13K/train.jsonl"
subset_path = args.small_dataset_file
subset_dir = os.path.dirname(subset_path)

if not os.path.exists(subset_dir):
    os.makedirs(subset_dir, exist_ok=True)
    print("create dir {}".format(subset_dir))

subset.to_json(subset_path)

# 加载新数据集, train是默认的split
loaded_dataset = load_dataset("json", data_files=subset_path, split="train")
print(loaded_dataset[0])
