BASE_DIR=/home/xiaojianha/labs/2504-longcot/Distill

# python dataset_downsize.py /home/xiaojianha/labs/Data/QwQ-LongCoT-130K_mmap \
#     --small_dataset_file ./QwQ-LongCoT-13K/train.jsonl
# dataset 220k
# python dataset_downsize.py /home/xiaojianha/labs/202504-longcot/Distill/open-r1-math-220k \
#     --small_dataset_file ./open-r1-math-22k/train.jsonl

# dataset_download
# python dataset_download.py simplescaling/s1K-1.1 \
#     --top_k 0 \
#     --output_dataset_path simplescaling/s1K-1.1/train.jsonl

# python dataset_download.py UWNSL/MATH_training_split_long_cot \
#     --top_k 0 \
#     --output_dataset_path ${BASE_DIR}/dataset/UWNSL/MATH_training_split_long_cot/train.jsonl

# python dataset_download.py UWNSL/MATH_training_split_short_cot \
#     --top_k 0 \
#     --output_dataset_path ${BASE_DIR}/dataset/UWNSL/MATH_training_split_short_cot/train.jsonl
 
python dataset_download.py Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B \
    --top_k 0 \
    --output_dataset_path ${BASE_DIR}/dataset/Magpie-Align/DeepDistill-250k/train.jsonl
