import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings

# --- Configuration ---
FILE_PATH = '/home/sowingg/coding/LLM/PPPUE/results/direct_eval_DP_eps200.0.json' # Please ensure the filename is correct

# Matplotlib global settings
warnings.filterwarnings('ignore')

# --- Helper Functions (No changes needed here) ---

def get_true_judgement(row):
    """
    Re-judges the prediction based on our own logic to fix the original Judgement bug.
    """
    true_label = row['True Profession'].lower().strip()
    gen_answer = str(row['Generated Answer']).lower().strip()

    # 1. Define synonyms
    synonyms = {
        'astronomer': ['astrophysicist', 'cosmologist'],
        'data scientist': ['data analyst'],
        'chef': ['cook'],
        'graphic designer': ['artist', 'visual designer'],
        'businessperson': ['retired ceo', 'ceo', 'business consultant', 'business development manager']
    }

    # 2. Check for garbage output
    if gen_answer == '[empty_output]' or all(c in '., ' for c in gen_answer):
        return False

    # 3. Check for direct match or synonym match
    if gen_answer == true_label:
        return True
    
    for key, values in synonyms.items():
        if (true_label == key and gen_answer in values) or \
           (gen_answer == key and true_label in values):
            return True
            
    return False

def is_garbage_output(answer):
    """
    Determines if the output is invalid content.
    """
    answer_norm = str(answer).lower().strip()
    return answer_norm == '[empty_output]' or not answer_norm or all(c in '., ' for c in answer_norm)

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
    sns.barplot(x=['Reported Accuracy', 'True Accuracy'], y=[reported_accuracy, true_accuracy], ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Reported vs. True Accuracy', fontsize=16)
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
    top_professions = df['True Profession'].value_counts().nlargest(7).index
    cm_df = df[df['True Profession'].isin(top_professions)]
    
    # Classify garbage and rare predictions as 'Other'
    valid_predictions = cm_df['Generated Answer'].value_counts().nlargest(7).index
    cm_df['Generated Answer Coded'] = cm_df['Generated Answer'].apply(lambda x: x if (x in valid_predictions and not is_garbage_output(x)) else 'Other')

    labels = sorted(list(set(cm_df['True Profession']) | set(cm_df['Generated Answer Coded'])))
    
    if len(labels) > 1:
        cm = confusion_matrix(cm_df['True Profession'], cm_df['Generated Answer Coded'], labels=labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix (True Profession vs. Generated Profession)', fontsize=18)
        plt.xlabel('Generated Profession', fontsize=14)
        plt.ylabel('True Profession', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
        print("Generated confusion matrix chart: confusion_matrix.png")
    
if __name__ == '__main__':
    analyze_experiment(FILE_PATH)