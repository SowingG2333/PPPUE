import os
import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

def resplit_data_grouped(input_file: str, output_dir: str, test_ratio: float = 0.1, random_seed: int = 42):
    """
    读取一个 JSONL 文件，根据 'personality' 字段进行分组，
    确保同一个 personality 的所有条目都在同一个数据集中（训练或测试），
    然后按比例切分为训练集和测试集。

    Args:
        input_file (str): 输入的 JSONL 文件路径。
        output_dir (str): 输出分割后文件的目录。
        test_ratio (float): 测试集所占的比例。训练集比例将是 1 - test_ratio。
        random_seed (int): 随机种子，确保每次切分结果一致。
    """
    print(f"--- 开始按 'personality' 分组处理数据集: {input_file} ---")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 按 personality 对所有数据进行分组
    grouped_data = defaultdict(list)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="正在读取和分组数据"):
                item = json.loads(line)
                # 将 personality 字典转换为可哈希的字符串作为 key
                personality_key = json.dumps(item['personality'], sort_keys=True)
                grouped_data[personality_key].append(item)
        
        num_total_samples = sum(len(v) for v in grouped_data.values())
        num_unique_personalities = len(grouped_data)
        print(f"成功读取 {num_total_samples} 条数据，分属于 {num_unique_personalities} 个独特的 'personality'。")
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"处理文件时出错: {e}")
        return

    # 2. 获取所有唯一的 personality key，并随机打乱这些 key
    personality_keys = list(grouped_data.keys())
    print(f"使用随机种子 {random_seed} 打乱 {len(personality_keys)} 个 personality 组...")
    random.seed(random_seed)
    random.shuffle(personality_keys)
    print("分组已打乱。")

    # 3. 按组分配到训练集和测试集
    train_keys = []
    test_keys = []
    
    # 计算测试集的组数
    test_group_size = int(len(personality_keys) * test_ratio)
    
    test_keys = personality_keys[:test_group_size]
    train_keys = personality_keys[test_group_size:]

    # 4. 根据分配好的 key 重新构建数据集
    train_data = []
    for key in train_keys:
        train_data.extend(grouped_data[key])
        
    test_data = []
    for key in test_keys:
        test_data.extend(grouped_data[key])

    print("\n--- 数据集划分统计 ---")
    print(f"总 'personality' 组数: {num_unique_personalities}")
    print(f"训练集组数: {len(train_keys)}")
    print(f"测试集组数: {len(test_keys)}")
    print("-" * 20)
    print(f"总样本数: {num_total_samples}")
    print(f"训练集样本数 (含验证集): {len(train_data)} ({len(train_data)/num_total_samples:.2%})")
    print(f"测试集样本数: {len(test_data)} ({len(test_data)/num_total_samples:.2%})")

    # 辅助函数：写入JSONL
    def write_jsonl(data_list, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 5. 写入文件
    # 根据您的要求，我们将训练集和验证集合并
    train_file_path = os.path.join(output_dir, "train_val_grouped.jsonl")
    test_file_path = os.path.join(output_dir, "test_grouped.jsonl")

    print("\n正在写入新文件...")
    write_jsonl(train_data, train_file_path)
    print(f"训练集 (含验证集) 已保存至: {train_file_path}")
    write_jsonl(test_data, test_file_path)
    print(f"测试集已保存至: {test_file_path}")
    
    print("\n--- 按 'personality' 分组划分完成！ ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按 'personality' 分组，将JSONL数据集重新划分为训练和测试集。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的原始JSONL文件路径。")
    parser.add_argument("--output_dir", type=str, required=True, help="存放新划分文件的输出目录。")
    parser.add_argument("--test_ratio", type=float, default=0.25, help="测试集比例，默认为 0.25。")
    parser.add_argument("--random_seed", type=int, default=42, help="用于数据打乱的随机种子，默认为 42。")
    
    args = parser.parse_args()
    
    print(f"注意：默认测试集比例为 {args.test_ratio:.0%}")
    
    resplit_data_grouped(
        input_file=args.input_file,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )