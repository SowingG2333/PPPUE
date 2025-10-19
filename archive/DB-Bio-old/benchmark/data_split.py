import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """从JSONL文件加载数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """将数据保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def split_dataset(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将数据集划分为训练集和测试集
    
    Args:
        data: 完整数据集
        train_ratio: 训练集比例 (默认0.8即80%)
        seed: 随机种子
    
    Returns:
        (train_data, test_data): 训练集和测试集
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)
    
    # 创建数据副本并打乱
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算分割点
    split_idx = int(len(shuffled_data) * train_ratio)
    
    # 划分数据集
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(
        description="将JSONL数据集划分为训练集和测试集"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入JSONL文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="输出目录路径 (默认为当前目录)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例 (默认0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认42)"
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default="train.jsonl",
        help="训练集文件名 (默认train.jsonl)"
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default="test.jsonl",
        help="测试集文件名 (默认test.jsonl)"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"正在从 '{args.input_file}' 加载数据...")
    data = load_jsonl(args.input_file)
    print(f"总共加载了 {len(data)} 条记录")
    
    # 划分数据集
    print(f"\n使用训练集比例 {args.train_ratio} 和随机种子 {args.seed} 进行划分...")
    train_data, test_data = split_dataset(
        data, 
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"训练集: {len(train_data)} 条记录 ({len(train_data)/len(data)*100:.1f}%)")
    print(f"测试集: {len(test_data)} 条记录 ({len(test_data)/len(data)*100:.1f}%)")
    
    # 保存数据集
    train_path = output_dir / args.train_name
    test_path = output_dir / args.test_name
    
    print(f"\n正在保存训练集到 '{train_path}'...")
    save_jsonl(train_data, str(train_path))
    
    print(f"正在保存测试集到 '{test_path}'...")
    save_jsonl(test_data, str(test_path))
    
    print("\n数据集划分完成!")
    
    # 显示统计信息
    print("\n=== 统计信息 ===")
    print(f"输入文件: {args.input_file}")
    print(f"总记录数: {len(data)}")
    print(f"训练集: {train_path} ({len(train_data)} 条)")
    print(f"测试集: {test_path} ({len(test_data)} 条)")

if __name__ == "__main__":
    main()