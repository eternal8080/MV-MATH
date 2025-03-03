import json
import re
import random
from openai import OpenAI
import httpx
from httpx_socks import SyncProxyTransport
from datetime import datetime 
import os 
import multiprocessing as mp

def get_model_answer(question, answer, standard_answer):
    # init DeepSeek API client
    client = OpenAI(api_key="api_key", base_url="https://api.deepseek.com")
    # use DeepSeek API extract aswers
    prompt_system = """
        You are a math expert. Given a question, a model's answer, and the standard answer, determine if the model's answer is correct. Only Respond with 'true' if the model's answer is correct, otherwise respond with 'false', without any additional information.
        For example:  
        Question: You are a expert in math problem solving.Structure Clauses: line Q R, line S T R, line Q P S, line P T\nSemantic Clauses: PT \\parallel QR\nIf ST = 8, TR = 4, and PT = 6, find QR. Choices are A: 6.0, B: 8.0, C: 9.0, D: 10.0. 
        Model Answer: The answer is C
        Standard Answer: 9.000
        Is the model's answer correct? true
        
        Question: <image>\nYou are a expert in math problem solving.Structure Clauses: line J I, line G H I, line G K J, line K H\nSemantic Clauses: KH \\parallel JI\nFind GI if GH = 9, GK = 6, and KJ = 4. Choices are A: 6.0, B: 9.0, C: 12.0, D: 15.0.
        Model Answer:Since KH is parallel to JI, we have angle GKJ = angle HGI. Also, since GH = 9, GK = 6, and KJ = 4, we can calculate that GH/GK = KJ/JI. Therefore, 9/6 = 4/JI. Solving for JI, we find that JI = 8. Thus, GI = GH - JI = 9 - 8 = 1. Therefore, the answer is D.\nAnswer:D
        Standard Answer: 15.000
        Is the model's answer correct? true
        
        Question: <image>\nYou are a expert in math problem solving.Structure Clauses: line P T, line S Q T, line S R, line P Q R\nSemantic Clauses: SR \\parallel PT, SQ = 3+x, TQ = 3, PQ = 6-x, RQ = 6+x\nFind PQ. Choices are A: 3.0, B: 4.0, C: 6.0, D: 9.0.
        Model Answer: The answer is B
        Standard Answer: 6.000"
        Is the model's answer correct? false
        
    """
    prompt_user = f"""
        Question: {question}
        Model Answer: {answer}
        Standard Answer: {standard_answer}
        Is the model's answer correct? Just respond with 'true' or 'false'.
    """
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]

    # call API for model answer
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=10,
        temperature=0.2,
        stream=False
    )

    # check if model answer is correct
    is_correct = response.choices[0].message.content.strip().lower()=="true"
    return is_correct

# process each problem
def process_problem(problem):
    answer = problem['model_answer']
    problem_id = problem['problem_id']
    question = problem['question']
    standard_answer = problem['standard_answer'].strip()

    # get model answer
    is_correct = get_model_answer(question, answer, standard_answer)  

    print(f"Processing problem ID: {problem_id}")
    print(f"Model Answer: {answer}")
    print(f"Is Correct: {is_correct}")
    
    return {
        "problem_id": problem_id,
        "question": question,
        "response": answer,
        "model_answer": answer,
        "standard_answer": standard_answer,
        "is_correct": is_correct
    }

def score_answers(jsonl_file, output_file):
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        problems = [json.loads(line) for line in file]

    with mp.Pool(16) as pool:  # use 16 processes
        results = pool.map(process_problem, problems)

    total_questions = len(results)
    correct_answers = sum(result['is_correct'] for result in results)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0

    summary = {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy
    }
    print(summary)
    results.append(summary)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    jsonl_file = "./outputs/model_results.jsonl"
    output_file = "./outputs/model_score.json"
    score_answers(jsonl_file, output_file)