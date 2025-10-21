import json
import random

SOURCE_JSONL = "/root/autodl-tmp/PPPUE/PersonalReddit/benchmark/train/api_anony_train_with_api_loss.jsonl"
TRAIN_JSONL = "/root/autodl-tmp/PPPUE/PersonalReddit/benchmark/train/seed_42/train_split.jsonl"
VAL_JSONL = "/root/autodl-tmp/PPPUE/PersonalReddit/benchmark/train/seed_42/val_split.jsonl"
VAL_RATIO = 0.2
SEED = 42

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    random.seed(SEED)

    samples = read_jsonl(SOURCE_JSONL)
    random.shuffle(samples)

    val_size = int(len(samples) * VAL_RATIO)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    write_jsonl(train_samples, TRAIN_JSONL)
    write_jsonl(val_samples, VAL_JSONL)

if __name__ == "__main__":
    main()