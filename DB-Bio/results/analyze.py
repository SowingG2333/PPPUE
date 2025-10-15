import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
import os
from openai import OpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed # 引入并行处理库

# --- Configuration ---
FILE_PATH = '/root/autodl-tmp/PPPUE/DB-Bio/results/new_strategy/eval_CLIPPING_ONLY.json' # 请确保文件名正确

# --- 新增：DeepSeek API 配置 ---
USE_DEEPSEEK_EVAL = True # 设置为 True 以启用 DeepSeek API 进行同义词判断
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY_HERE") # 优先从环境变量读取
MAX_WORKERS = 20 # 设置并行API请求的工作线程数

# 初始化 DeepSeek 客户端
client = None
if USE_DEEPSEEK_EVAL:
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_DEEPSEEK_API_KEY_HERE":
        print("DeepSeek API key found. Initializing client.")
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    else:
        print("Warning: DEEPSEEK_API_KEY not set. Falling back to strict matching.")
        USE_DEEPSEEK_EVAL = False

# Matplotlib global settings
warnings.filterwarnings('ignore')
# tqdm.pandas(desc="Evaluating with API") # 不再需要 pandas 的 tqdm 集成

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

def get_deepseek_judgement(true_label, gen_answer, client_instance):
    """
    使用 DeepSeek API 判断预测是否正确（严格匹配或同义词）。
    此函数现在接收单独的字符串参数，以便于并行化。
    """
    true_label = true_label.lower().strip()
    gen_answer = str(gen_answer).lower().strip()

    # 1. 检查垃圾输出
    if is_garbage_output(gen_answer):
        return False

    # 2. 检查严格匹配，如果匹配成功则直接返回 True，节省 API 调用
    if gen_answer == true_label:
        return True

    # 3. 调用 DeepSeek API 进行同义词判断
    try:
        prompt = f"""
        你是一个职业领域的专家。请判断以下两个职业名称是否指代同一个职业或互为同义词。
        职业A："{true_label}"
        职业B："{gen_answer}"
        请只回答“是”或“否”。
        """
        response = client_instance.chat.completions.create(
            model="deepseek-chat", # 或者使用 deepseek-coder
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        api_result = response.choices[0].message.content.strip()
        time.sleep(0.1) # 并行时可以适当减少休眠时间
        return "是" in api_result
    except Exception as e:
        print(f"API call failed for '{true_label}' vs '{gen_answer}': {e}. Defaulting to False.")
        return False


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
    print("\n--- Re-evaluating predictions ---")
    if USE_DEEPSEEK_EVAL and client:
        print(f"Using DeepSeek API for synonym-aware evaluation with {MAX_WORKERS} workers...")
        results = []
        # 使用 ThreadPoolExecutor 进行并行处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 创建一个 future 到 index 的映射，以便按原始顺序排序结果
            future_to_index = {
                executor.submit(get_deepseek_judgement, row['True Profession'], row['Generated Answer'], client): index
                for index, row in df.iterrows()
            }
            
            # 使用 tqdm 创建进度条
            pbar = tqdm(as_completed(future_to_index), total=len(df), desc="Evaluating with API")
            
            # 收集结果
            temp_results = {}
            for future in pbar:
                index = future_to_index[future]
                try:
                    result = future.result()
                    temp_results[index] = result
                except Exception as exc:
                    print(f'Row {index} generated an exception: {exc}')
                    temp_results[index] = False # 出错时默认为 False
            
            # 按原始索引排序结果
            results = [temp_results[i] for i in range(len(df))]

        df['True Judgement'] = results
    else:
        print("Using strict string matching evaluation...")
        df['True Judgement'] = df.apply(get_true_judgement, axis=1)

    df['Is Garbage'] = df['Generated Answer'].apply(is_garbage_output)

    # 3. Calculate core metrics
    reported_accuracy = metrics.get('accuracy', 0)
    true_accuracy = df['True Judgement'].mean()
    garbage_percentage = df['Is Garbage'].mean()

    print("\n--- Core Metrics Analysis ---")
    eval_method = "DeepSeek Synonym-Aware" if USE_DEEPSEEK_EVAL else "Strict Match"
    print(f"Evaluation Method: {eval_method}")
    print(f"Reported Accuracy (Potentially Bugged): {reported_accuracy:.2%}")
    print(f"Corrected True Accuracy: {true_accuracy:.2%}")
    print(f"Invalid Output (Empty or repeated symbols) Percentage: {garbage_percentage:.2%}")

    # 4. Analyze error patterns
    incorrect_df = df[~df['True Judgement']]
    
    # Count the most common incorrect predictions (excluding garbage output)
    incorrect_valid_predictions = incorrect_df[~incorrect_df['Is Garbage']]['Generated Answer'].value_counts()
    
    # Calculate model performance on different professions
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

    # Plot 1: True Accuracy vs. Reported Accuracy
    accuracy_label = 'True Accuracy (Synonym Aware)' if USE_DEEPSEEK_EVAL else 'True Accuracy (Strict)'
    sns.barplot(x=['Reported Accuracy', accuracy_label], y=[reported_accuracy, true_accuracy], ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Reported vs. Corrected Accuracy', fontsize=16)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    for p in axes[0, 0].patches:
        axes[0, 0].annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Plot 2: Invalid Output Percentage (Robust pie chart)
    garbage_counts = df['Is Garbage'].value_counts()
    pie_data = []
    pie_labels = []
    pie_colors = []

    if False in garbage_counts.index:
        pie_data.append(garbage_counts[False])
        pie_labels.append('Valid Output')
        pie_colors.append('#86BBD8')
    
    if True in garbage_counts.index:
        pie_data.append(garbage_counts[True])
        pie_labels.append('Invalid Output')
        pie_colors.append('#F26419')

    if pie_data: # Ensure there is data to plot
        axes[0, 1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors, textprops={'fontsize': 12})
    axes[0, 1].set_title('Output Validity Analysis', fontsize=16)

    # Plot 3: Top 10 Incorrect Predictions (non-garbage)
    top_n = 10
    top_incorrect = incorrect_valid_predictions.head(top_n)
    if not top_incorrect.empty:
        sns.barplot(x=top_incorrect.values, y=top_incorrect.index, ax=axes[1, 0], palette='coolwarm')
        axes[1, 0].set_title(f'Top {top_n} Incorrect Predictions (Model\'s Bias)', fontsize=16)
        axes[1, 0].set_xlabel('Number of Errors', fontsize=12)

    # Plot 4: Accuracy per profession
    if not accuracy_per_profession.empty:
        sns.barplot(x=accuracy_per_profession.values, y=accuracy_per_profession.index, ax=axes[1, 1], palette='mako')
        axes[1, 1].set_title('Model Performance on Professions (Knowledge Gaps)', fontsize=16)
        axes[1, 1].set_xlabel('Accuracy', fontsize=12)
        axes[1, 1].set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("analysis_summary.png", dpi=300, bbox_inches='tight')
    print("\nGenerated analysis chart: analysis_summary.png")

    # Confusion Matrix
    # For clarity, we only show the most common professions
    
    # 标准化职业名称（忽略大小写）
    df['True Profession Normalized'] = df['True Profession'].str.lower().str.strip()
    df['Generated Answer Normalized'] = df['Generated Answer'].str.lower().str.strip()
    
    top_professions = df['True Profession Normalized'].value_counts().nlargest(7).index
    cm_df = df[df['True Profession Normalized'].isin(top_professions)].copy()
    
    # Classify garbage and rare predictions as 'Other'
    valid_predictions = cm_df['Generated Answer Normalized'].value_counts().nlargest(7).index
    cm_df['Generated Answer Coded'] = cm_df['Generated Answer Normalized'].apply(
        lambda x: x if (x in valid_predictions and not is_garbage_output(x)) else 'other'
    )

    labels = sorted(list(set(cm_df['True Profession Normalized']) | set(cm_df['Generated Answer Coded'])))
    
    if len(labels) > 1:
        cm = confusion_matrix(cm_df['True Profession Normalized'], cm_df['Generated Answer Coded'], labels=labels)
        
        # 创建更大的图形以适应标签
        fig_size = max(14, len(labels) * 1.5)
        plt.figure(figsize=(fig_size, fig_size * 0.85))
        
        # 使用更好的颜色方案和格式
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Number of Samples', 'shrink': 0.75, 'pad': 0.02},
                    linewidths=0.8, linecolor='white',
                    square=True,
                    annot_kws={'fontsize': 11, 'weight': 'bold'},
                    vmin=0)  # 确保颜色从0开始
        
        plt.title('Confusion Matrix (True Profession vs. Generated Profession)', 
                  fontsize=20, fontweight='bold', pad=25)
        plt.xlabel('Predicted Profession', fontsize=15, fontweight='bold', labelpad=12)
        plt.ylabel('True Profession', fontsize=15, fontweight='bold', labelpad=12)
        
        # 优化标签显示
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=12)
        plt.yticks(rotation=0, fontsize=12, va='center')
        
        # 调整布局以防止标签被裁剪
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print("Generated confusion matrix chart: confusion_matrix.png")
    
if __name__ == '__main__':
    analyze_experiment(FILE_PATH)