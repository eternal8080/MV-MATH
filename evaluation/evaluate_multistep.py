import argparse
import json
import os
import multiprocessing as mp
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

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
    
    print(f"Processing problem ID: {problem_id}, {is_correct}")
    return {
        "problem_id": problem_id,
        "question_translate": standard_info.get("translate_question", ""),
        "input_image": standard_info.get("input_image", []),
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
def filter_and_extract_answers_parallel(input_file, output_file, standard_file, num_processes=4):
    ensure_output_dir(output_file)

    standard_answers = load_standard_answers(standard_file)

    # load response file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    problems = [json.loads(line.strip()) for line in lines]

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(process_problem, [(problem, standard_answers) for problem in problems]),
                            total=len(problems), desc="Processing problems"))

   # 统计正确步骤总数和步骤总数 + 完整正确题目的数量
    total_correct_steps = 0
    total_steps = 0
    complete_correct_questions = 0  # 完全正确题目计数

    for result in results:
        is_correct = result["is_correct"]
        if is_correct:
            try:
                correct, steps = map(int, is_correct.split('/'))
                total_correct_steps += correct
                total_steps += steps
                if correct == steps:
                    complete_correct_questions += 1
            except ValueError:
                print(f"Invalid is_correct format in result: {result['problem_id']} -> {is_correct}")

    Step_Accuracy_Rate = total_correct_steps / total_steps if total_steps > 0 else 0
    Question_Completeness_Rate = complete_correct_questions / len(results) if results else 0


    # 生成最终结果并写入输出文件
    summary = {
        "total_questions": len(results),
        "total_correct_steps": total_correct_steps,
        "total_steps": total_steps,
        "Step_Accuracy_Rate": Step_Accuracy_Rate,
        "Question_Completeness_Rate": Question_Completeness_Rate,
    }
    results.append(summary)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=4)

    print(f"saved to {output_file}")
    print(summary)

# check_answer_correctness
def check_answer_correctness(standard_answer, response):
    prompt_system = """
    You are a math expert. You need to check whether the model's final answer is consistent with the standard answer. Your answer is in the form of correct step/total step. For example, if there are three questions in the question, and the model correctly answers 2 questions, then the output is 2/3. If the model correctly answers 0 questions, then the output is 0/3.
    """

    prompt_user = f"""
                    Standard answer: {standard_answer}
                    Model's response: {response}
                    Only output correct step/total step, without any other steps. 
                    """

    messages = [{
        "role": "system", 
        "content": prompt_system},
                {
                      "role": "user",
                      "content": prompt_user
                     }]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        stream=False
    )

    model_answer = response.choices[0].message.content.strip().lower()
    return model_answer  


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the answers JSONL file.")
    parser.add_argument("--response-file", type=str, help="Path to the response JSONL file.", default="Path/to/response/JSONL")
    parser.add_argument("--output-file", type=str, help="Path to the output comparison JSON file.", default="Path/to/output/JSON")
    parser.add_argument("--standard-file", type=str, help="Path to the standard answers JSON file.", default="Path/to/standard/JSON")
    args = parser.parse_args()

    filter_and_extract_answers_parallel(args.response_file, args.output_file, args.standard_file, num_processes=16)
