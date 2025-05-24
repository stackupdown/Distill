from datasets import load_dataset, concatenate_datasets
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

def prepare_data(all_set_path, top_k, output_dataset_path):
    dataset = load_dataset(all_set_path, split="train")

    if top_k:
        subset = dataset.select(range(args.top_k))
    else:
        subset = dataset

    subset_dir = os.path.dirname(output_dataset_path)

    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir, exist_ok=True)
        print("create dir {}".format(subset_dir))
    return dataset

# dataset = prepare_data(args.dataset_path, args.top_k, args.output_dataset_path)
dataset = prepare_data('Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B', 0,
                       'dataset/Magpie-Align/DeepDistill-250k')

def filter_difficulty(dataset, func, count):
    remove_names = set(dataset.column_names)
    remove_names -= {'response', 'difficulty', 'instruction', 'conversation_id', 'task_category'}
    dataset = dataset.filter(func).shuffle(seed=42)
    if count == 0:
        return dataset.map(
            lambda x: x
        )
    dataset = dataset.select(range(count)).map(
        lambda x: x, remove_columns=remove_names
    )
    return dataset


# 11853, 141300, 79983 = 1:14:8
def merge_deepstill(dataset):
    # Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B
    # # 加载新数据集, train是默认的split
    # task_category
    # 类别: Advice seeking, 出现次数: 163
    # 类别: Brainstorming, 出现次数: 26
    # 类别: Coding & Debugging, 出现次数: 1508
    # 类别: Creative writing, 出现次数: 9
    # 类别: Data analysis, 出现次数: 4201
    # 类别: Editing, 出现次数: 106
    # 类别: Information seeking, 出现次数: 3777
    # 类别: Math, 出现次数: 234544
    # 类别: Planning, 出现次数: 1553
    # 类别: Reasoning, 出现次数: 4018
    # 类别: Role playing, 出现次数: 7
    # easy, hard, medium, very easy, very hard
    all_datasets = []
    x = filter_difficulty(
        dataset, lambda x: x['task_category'] == 'Math' and x['difficulty'] == 'hard', 500)
    all_datasets.append(x)
    x = filter_difficulty(
        dataset, lambda x: x['task_category'] == 'Math' and x['difficulty'] == 'medium', 2500)
    all_datasets.append(x)
    x = filter_difficulty(
        dataset, lambda x: x['task_category'] == 'Math' and x['difficulty'] == 'easy', 3500)
    all_datasets.append(x)
    x = filter_difficulty(
        dataset, lambda x: x['task_category'] == 'Reasoning' and x['difficulty'] in ['medium', 'easy'], 3000
    )
    all_datasets.append(x)

    x = filter_difficulty(
        dataset, lambda x: x['task_category'] == 'Data analysis' and x['difficulty'] == 'medium', 500
    )

    all_datasets.append(x)

    result = concatenate_datasets(all_datasets)
    result.to_json('dataset/Magpie-Align/DeepDistill-250k/train.jsonl')

merge_deepstill(dataset)

def load_dataset_test(subset_path):
    loaded_dataset = load_dataset("json", data_files=subset_path, split="train")
    print(loaded_dataset[0])
