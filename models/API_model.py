import requests
import json
import subprocess
import argparse 
import base64
import time
import datetime
from openai import OpenAI
import os

def string_to_dict(json_string):
    try:
        dictionary = json.loads(json_string)
        return dictionary
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
# from prompt.system_prompt import system_prompt
def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        base64_datas = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_datas}"

url_map = {
    "gpt4v": "",
    "gemini-1.5-pro": "",
    "claude-3.5-sonnet": "",
    "gpt4o": ""
}

api_keys_map = {
    "gpt4v": "",
    "gpt4o": "",
    "gemini-1.5-pro": "",
    "claude-3.5-sonnet": ""
}

class StatusCodeError(Exception):
    pass

def get_chat_vllm_response(args, vllm_model, question, ocr_result, sampling_parameter):
    # import pdb; pdb.set_trace()
    # if args.use_ocr:
    #     prompt = [q+assist_ocr_prompt.format(ocr) for q, ocr  in  zip(question, ocr_result)]
    # else:
    prompt =  question
    result = vllm_model.generate(prompt, sampling_parameter)
    return [result[i].outputs[0].text for i in range(len(result))]


def get_chat_response_batch_text(args, questions, images):
    def get_chat_response_text(args, question, images):
        sleep_time = 1
        patience = 20
        max_tokens = 1024
        temperature = 0.2
        invalid_response = ['']

        while patience > 0:
            patience -= 1
            try:
                # print("self.model", self.model)
                url = url_map[args.model_name]
                apiKey = api_keys_map[args.model_name] 
                headers = {
                    'api-key': apiKey,   
                    "Content-Type": "application/json; charset=utf-8",
                    "Encoding": "utf-8"
                }
                base64_images = [image_to_base64(image) for image in images]
                base64_images = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}"
                        },
                    }
                    for base64_image in base64_images
                ]
                # import pdb; pdb.set_trace()
                if args.model_name=="gpt4v" or args.model_name=="gpt4o":
                    system_content = {
                                        "role": "system",
                                        "content": "You are a math problem-solving assistant. Your input is a math problem and the images of the problem. Your task is to output the solution ideas and answers to the problem. The output format is a step-by-step approach. Each question is a multiple-choice question, and you need to enter the correct option at the end. There is only one correct option for each question. Put your final answer within {}. Your final answer must be one of A, B, C, and D."
                                    }
                    question_content = [
                                    {
                                    "type": "text",
                                    "text": question
                                    }
                                    ]
                    question_content.extend(base64_images)
                    user_content = {
                                    "role":"user",
                                    "content": question_content
                                    }
                    
                    data = {
                            "messages": [
                                system_content,
                                user_content
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    # import pdb; pdb.set_trace()
                    response = requests.post(url, headers=headers, json=data)
                    if args.model_name == "gpt4v" or "gpt4o":
                        if response.status_code == 200:
                            invalid_response = response.json()["choices"][0]["message"]["content"]
                            # print('-'*30, invalid_response)
                            if invalid_response != '':
                                return response.json()["choices"][0]["message"]["content"]
                            else:
                                print('response is invalid')
                            # raise StatusCodeError(f"Expected status code 200 but received {response.status_code}")
                        else:
                            raise StatusCodeError(f"Expected status code 200 but received {response.status_code}")
                    else:
                        ValueError("Fault Model Name")
                if args.model_name=="claude-3.5-sonnet" or "gemini-1.5-pro":
                    # import pdb; pdb.set_trace()
                    def get_response():
                        client = OpenAI(
                            api_key=apiKey, # os.getenv("TAL_MLOPS_APP_KEY"),  
                            base_url=url,  
                        )
                        question_content = [
                                        {
                                        "type": "text",
                                        "text": question
                                        }
                                        ]
                        question_content.extend(base64_images)
                        user_content = {
                                        "role":"user",
                                        "content": question_content
                                        }
                        
                        completion = client.chat.completions.create(
                            model=args.model_name,
                            messages=[
                            {
                                "role":"system",
                                "content":"You are a math problem-solving assistant. Your input is a math problem and the images of the problem. Your task is to output the solution ideas and answers to the problem. The output format is a step-by-step approach. Each question is a multiple-choice question, and you need to enter the correct option at the end. There is only one correct option for each question. put your final answer within {}. Your final answer must be one of A, B, C, and D."   # 这里是你的system prompt
                            },
                            user_content,
                            ],
                            max_tokens = max_tokens,
                            temperature = temperature
                            )
                        # import pdb; pdb.set_trace()
                        return string_to_dict(completion.json())["choices"][0]["message"]["content"]
                    return get_response()
        
                    
            except Exception as e:
                if "limit" not in str(e):
                    print(e)
                if "Please reduce the length of the messages or completion" in str(e):
                    max_tokens = int(max_tokens * 0.9)
                    print("!!Reduce max_tokens to", max_tokens)
                if "Please reduce the length of the messages." in str(e):
                    print("!!Reduce user_prompt to", user_prompt[:-1])
                    return ""
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "" 

    results = []
    for question in questions:
        # print(question)
        results.append(get_chat_response_text(args, question, images))
    return results


def load_train_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# Two-shot 
two_shot_examples = """
Question: If a triangle has two sides of length 3 and 4, what is the length of the hypotenuse? A.10 B.8 C.5 D.4

Answer:
Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:
\[c^2 = a^2 + b^2\], where \( a \) and \( b \) are the legs, and \( c \) is the hypotenuse.
Step 2 (Substitute the known values): Given \( a = 3 \) and \( b = 4 \). Substituting these values into the formula:
\[c^2 = 3^2 + 4^2 = 9 + 16 = 25\]
Step 3 (Calculate the hypotenuse): Taking the square root gives:
\[c = \sqrt{25} = 5\]
Answer: {C}

Question: In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. Find the area of the right triangle. A.20 B.10 C.5 D.6
Answer:
Step 1 (Mathematical theorem used: Pythagorean theorem): We first use the Pythagorean theorem to find the length of \( BC \). The formula is:
\[AC^2 = AB^2 + BC^2\], where \( AC \) is the hypotenuse, and \( AB \) and \( BC \) are the legs.
Step 2 (Substitute the known values): Given \( AC = 5 \) and \( AB = 4 \). Substituting these values:
\[5^2 = 4^2 + BC^2 \implies 25 = 16 + BC^2\]
Step 3 (Solve for \( BC \)):
\[BC^2 = 25 - 16 = 9 \implies BC = \sqrt{9} = 3\]
Step 4 (Calculate the area): The area of the right triangle is given by \( \frac{1}{2} \times AB \times BC \). Substituting the known values:
\[\text{Area} = \frac{1}{2} \times 4 \times 3 = 6\]
Answer: {D}

Your final answer must be one of A, B, C, and D.
\nPlease reason step by step, and put your final answer within {}.Each step is placed on a new line, using the following format: Step X (Mathematical theorem/basis used): Detailed solution steps. Answer: {}
"""
if __name__ == "__main__":
    train_data = load_train_data(r'your/data/path')
    image_folder = r"your/image/folder"

    all_response = []
    i = 0
    # for question_id, question in train_data.items():
    for prob in train_data:
        question_id = prob["problem_id"]
        question = prob["translate_question"]
        images = [os.path.join(image_folder, img).replace("\\", "/") for img in prob["input_image"]]
        
        # i += 1
        # if i == 2:
        #     break
    
        prompt = f"{two_shot_examples}\n Question:{question}"
        question_input = [prompt + question]
        # print(question_input)
        import argparse
        # define a dictionary
        dict_args = {'model_name': 'claude-3.5-sonnet'}
        parser = argparse.ArgumentParser()

        for key, value in dict_args.items():
            parser.add_argument('--' + key, default=value)

        args = parser.parse_args()
        response = get_chat_response_batch_text(args, question_input, images)
        print(question_id, response)
        all_response.append({
            "problem": question_id,
            "response": response[0]
        })

        # time.sleep(5)

    filename = r"your/output/json/file"


with open(filename, 'w', encoding='utf-8') as file:
    json.dump(all_response, file, ensure_ascii=False, indent=4)

    print(f"written to {filename}")



    