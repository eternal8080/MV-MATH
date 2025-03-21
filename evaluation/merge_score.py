import json
import os

# Automatically detect the type of evaluation result by inspecting its fields
def detect_type_and_load(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # If the file is a list (e.g., multi-step results), extract the summary part
        if isinstance(data, list):
            data = data[-1]

        # Identify the type of the result based on its keys
        if "total_correct_steps" in data:
            return "multi_step", data
        elif "correct_answers" in data and data["total_questions"] < 1000:
            return "free_form", data
        elif "correct_answers" in data:
            return "multiple_choice", data
        else:
            return "unknown", data

# Load all result files and merge them into a unified dictionary
def merge_results(json_files):
    summary = {}
    for file in json_files:
        type_name, result_data = detect_type_and_load(file)
        if type_name == "unknown":
            print(f"⚠️ Unrecognized file format: {file}")
        else:
            summary[type_name] = result_data
    return summary

# Compute the weighted score based on correct answers or complete correctness
def compute_weighted_score(summary):
    total_correct = 0
    total_questions = 0

    # For free-form answers: use correct_answers field
    if "free_form" in summary:
        total_correct += summary["free_form"].get("correct_answers", 0)
        total_questions += summary["free_form"].get("total_questions", 0)

    # For multiple choice: use correct_answers field
    if "multiple_choice" in summary:
        total_correct += summary["multiple_choice"].get("correct_answers", 0)
        total_questions += summary["multiple_choice"].get("total_questions", 0)

    # For multi-step: use completely correct questions (e.g., 3/3)
    if "multi_step" in summary:
        total_correct += summary["multi_step"].get("complete_correct_questions", 0)
        total_questions += summary["multi_step"].get("total_questions", 0)

    weighted_score = total_correct / total_questions if total_questions > 0 else 0.0
    return {
        "total_questions_all": total_questions,
        "total_correct_all": total_correct,
        "weighted_score": weighted_score
    }

if __name__ == "__main__":
    # You can modify the following file paths
    files = [
        "path/to/evaluate/multi_step.json",
        "path/to/evaluate/free_form.json",
        "path/to/evaluate/multiple_choice.json"
    ]

    final_summary = merge_results(files)

    # Compute weighted accuracy
    weighted_summary = compute_weighted_score(final_summary)
    final_summary["weighted_summary"] = weighted_summary

    # Save the merged summary into a new JSON file
    with open("final_summary.json", "w", encoding='utf-8') as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    # Print weighted score for quick reference
    print("✅ Summary complete. Saved as final_summary.json")
    print(f"\n📊 Weighted Score: {weighted_summary['weighted_score']:.4f} "
          f"({weighted_summary['total_correct_all']}/{weighted_summary['total_questions_all']})")
