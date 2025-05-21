from datasets import load_dataset, Dataset, concatenate_datasets

# 加载第一个数据集并取前1000条
def load_first_dataset():
    qpath = 'qihoo360/Light-R1-SFTData'
    qdataset = load_dataset(qpath, split='train').select(range(1200))
    length = 0
    for i in range(len(qdataset)):
        item = qdataset[i]
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        length += len(answer)

    qdataset = qdataset.map(
        lambda x: {'question': x['conversations'][0]['value'], 'answer': x['conversations'][1]['value'],
                   'type': 'long'}, remove_columns=['conversations']
    )

    print(qdataset.column_names)
    return qdataset

# 加载第二个数据集（修正后的正确路径）
def load_second_dataset():
    path = 'UWNSL/MATH_training_split_short_cot'
    # 5383
    short_dataset = load_dataset(path, split='train').select(range(4800))
    length = 0
    for i in range(len(short_dataset)):
        item = short_dataset[i]
        length += len(item['solution'])
    short_dataset = short_dataset.map(
        lambda x: {'question': x['problem'], 'answer': '<think>' + x['solution'] + '</think>',
                   'type': 'short'},
        remove_columns=short_dataset.column_names
    )
    print(short_dataset.column_names)
    return short_dataset

qdataset = load_first_dataset()
short_dataset = load_second_dataset()
# 合并数据集
combined_dataset = concatenate_datasets([qdataset, short_dataset])

# 打乱顺序（设置随机种子保证可重复性）
shuffled_dataset = combined_dataset.shuffle(seed=42)

# 保存到本地
save_path = './mixed_dataset/mixed_short_lightr1_dataset.json'
shuffled_dataset.to_json(save_path)

print(f"数据集已保存至：{save_path}")
print(f"最终数据集大小：{len(shuffled_dataset)} 条")

dataset = load_dataset('json', data_files=save_path, split='train')
print(dataset.column_names)
