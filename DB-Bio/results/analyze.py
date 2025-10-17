import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# --- Configuration ---
FILE_PATH = ''

# Matplotlib global settings
warnings.filterwarnings('ignore')

# --- Helper Functions (No changes needed here) ---

def get_true_judgement(row):
    """
    严格匹配判断：仅当预测职业与真实职业完全一致时才判定为正确（忽略大小写）
    """
    true_label = row['True Profession'].lower().strip()
    gen_answer = str(row['Generated Answer']).lower().strip()

    # 1. 检查垃圾输出
    if is_garbage_output(gen_answer):
        return False

    # 2. 仅进行严格的完全匹配（已转换为小写，忽略大小写差异）
    return gen_answer == true_label

def is_garbage_output(answer):
    """
    判断输出是否为无效内容（更严格的检测）
    """
    answer_norm = str(answer).lower().strip()
    
    # 检查空输出或标记
    if answer_norm == '[empty_output]' or not answer_norm:
        return True
    
    # 检查是否只包含标点符号和空格
    if all(c in '., \n\t' for c in answer_norm):
        return True
    
    # 检查是否是重复字符（如 "..." 或 "。。。"）
    if len(set(answer_norm.replace(' ', ''))) <= 2 and len(answer_norm) > 5:
        return True
    
    # 检查是否包含至少一个字母或汉字
    has_valid_char = any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in answer_norm)
    if not has_valid_char:
        return True
        
    return False

# --- Main Analysis Workflow ---

def analyze_experiment(file_path):
    """Main analysis function"""
    print(f"--- Analyzing experiment file: {file_path} ---")

    # 1. Load data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data['qualitative_samples'])
    metrics = data.get('metrics', {}) # Use .get for robustness

    # 2. Preprocess and re-evaluate
    df['True Judgement'] = df.apply(get_true_judgement, axis=1)
    df['Is Garbage'] = df['Generated Answer'].apply(is_garbage_output)

    # 3. Calculate core metrics
    reported_accuracy = metrics.get('accuracy', 0)
    true_accuracy = df['True Judgement'].mean()
    garbage_percentage = df['Is Garbage'].mean()

    print("\n--- Core Metrics Analysis ---")
    print(f"Reported Accuracy (Potentially Bugged): {reported_accuracy:.2%}")
    print(f"Corrected True Accuracy: {true_accuracy:.2%}")
    print(f"Invalid Output (Empty or repeated symbols) Percentage: {garbage_percentage:.2%}")

    # 4. Analyze error patterns
    incorrect_df = df[~df['True Judgement']]
    incorrect_valid_predictions = incorrect_df[~incorrect_df['Is Garbage']]['Generated Answer'].value_counts()
    accuracy_per_profession = df.groupby('True Profession')['True Judgement'].mean().sort_values(ascending=False)

    print("\n--- Prediction Behavior Analysis ---")
    print("Top 5 Most Common Valid Incorrect Predictions:")
    print(incorrect_valid_predictions.head(5))
    print("\nTrue Accuracy Per Profession:")
    print(accuracy_per_profession)

    # --- 5. Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle('In-depth Analysis of Experiment Results', fontsize=20, y=1.02)
    sns.set_style("whitegrid")

    # Plot 1: Profession Distribution
    profession_counts = df['True Profession'].value_counts()
    sns.barplot(x=profession_counts.index, y=profession_counts.values, ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Profession Distribution', fontsize=16)
    axes[0, 0].set_ylabel('Number of Samples', fontsize=12)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

    # Plot 2: Output Type Distribution (Correct/Incorrect/Garbage Output)
    error_types = pd.Series(np.where(df['True Judgement'], 'Correct', np.where(df['Is Garbage'], 'Garbage Output', 'Incorrect'))).value_counts()
    sns.barplot(x=error_types.index, y=error_types.values, ax=axes[0, 1], palette='Set1')
    axes[0, 1].set_title('Output Type Distribution', fontsize=16)
    axes[0, 1].set_ylabel('Number of Samples', fontsize=12)

    # Plot 3: 准确率对比图，图片上显示修正前后准确率
    axes[1, 0].bar(['Reported Accuracy', 'Corrected True Accuracy'], [reported_accuracy, true_accuracy], color=['skyblue', 'salmon'])
    axes[1, 0].set_title('Accuracy Comparison', fontsize=16)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate([reported_accuracy, true_accuracy]):
        axes[1, 0].text(i, v + 0.02, f"{v:.2%}", ha='center', fontsize=12)

    # Plot 4: Model Accuracy by Profession
    if not accuracy_per_profession.empty:
        sns.barplot(x=accuracy_per_profession.values, y=accuracy_per_profession.index, ax=axes[1, 1], palette='mako')
        axes[1, 1].set_title('Model Accuracy by Profession', fontsize=16)
        axes[1, 1].set_xlabel('Accuracy', fontsize=12)
        axes[1, 1].set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("analysis.png", dpi=300, bbox_inches='tight')
    print("\nGenerated analysis chart: analysis.png")

if __name__ == '__main__':
    analyze_experiment(FILE_PATH)