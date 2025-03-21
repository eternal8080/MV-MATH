import json
import os
import argparse
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import multiprocessing as mp

client = OpenAI(api_key="api_key", base_url="https://api.deepseek.com")

# ensure_output_dir 
def ensure_output_dir(output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# call API for model answer
def process_problem(problem, standard_answers):
    response = problem['model_answer']
    problem_id = problem['problem_id']
    standard_info = standard_answers.get(problem_id, {})
    standard_answer = standard_info.get("answer", "").strip()

    is_correct = check_answer_correctness(standard_answer, response)
    
    return {
        "problem_id": problem_id,
        "response": response,
        "standard_answer": standard_answer,
        "is_correct": is_correct
    }

# load standard answers
def load_standard_answers(standard_file):
    with open(standard_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        item["problem_id"]: {
            "answer": item["answer"].strip(),
            "translate_question": item.get("translate_question", ""),
            "input_image": item.get("input_image", "")
        } for item in data
    }

# filter_and_extract_answers_parallel
def filter_and_extract_answers_parallel(response_file, output_file, standard_file, num_processes=8):
    ensure_output_dir(output_file)

    standard_answers = load_standard_answers(standard_file)

    # load response file
    with open(response_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    problems = [json.loads(line.strip()) for line in lines]

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(process_problem, [(problem, standard_answers) for problem in problems]),
                            total=len(problems), desc="Scoring problems"))

    total_questions = len(results)
    correct_answers = sum(1 for result in results if result["is_correct"])
    accuracy = correct_answers / total_questions if total_questions > 0 else 0

    summary = {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy
    }
    results.append(summary)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=4)

    print(f"saved to {output_file}")
    print(summary)

# check_answer_correctness
def check_answer_correctness(standard_answer, response):
   # prompt the user to check if the model's answer is correct
    prompt_system = f"""
                    You are a math expert. You need to check if the model's final answer matches the standard answer.
                    """

    prompt_user = f"""
                    Standard answer: {standard_answer}
                    Model's response: {response}
                    Do they match? Just answer 'true' if correct, 'false' if incorrect.
                    """
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        stream=False
    )

    model_answer = response.choices[0].message.content.strip().lower()
    return model_answer == "true"

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the answers JSONL file.")
    parser.add_argument("--response-file", type=str, help="Path to the response JSONL file.", default="Path/to/response/freeform/JSONL")
    parser.add_argument("--output-file", type=str, help="Path to the output evaluation JSON file.", default="Path/to/output/evaluation/freeform/JSON")
    parser.add_argument("--standard-file", type=str, help="Path to the standard answers JSON file.", default="Path/to/standard/JSON")
    args = parser.parse_args()


    # score the answers
    filter_and_extract_answers_parallel(args.response_file, args.output_file, args.standard_file, num_processes=16)
