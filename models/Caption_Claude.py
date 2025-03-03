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
    "gpt4v": "xxx",
    "gemini-1.5-pro": "xxx",
    "claude-3.5-sonnet": "xxx",
    "gpt4o": "xxx"
}

api_keys_map = {
    "gpt4v": "xxx",
    "gpt4o": "xxx",
    "gemini-1.5-pro": "xxx",
    "claude-3.5-sonnet": "xxx"
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
        # model = "gpt-4-turbo-vision" if args.model_name=="gpt4v" else "gpt4o"
        # import pdb; pdb.set_trace()
        while patience > 0:
            patience -= 1
            try:
                # print("self.model", self.model)
                url = url_map[args.model_name]
                apiKey = api_keys_map[args.model_name] # your api key
                headers = {
                    'api-key': apiKey,   # your api key
                    "Content-Type": "application/json; charset=utf-8",
                    "Encoding": "utf-8"
                }
                base64_images = [image_to_base64(images)]
                base64_images = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{base64_image}"
                        },
                    }
                    for base64_image in base64_images
                ]
                if args.model_name=="claude-3.5-sonnet" or "gemini-1.5-pro":
                    # import pdb; pdb.set_trace()
                    def get_response():
                        client = OpenAI(
                            api_key=apiKey, # os.getenv("TAL_MLOPS_APP_KEY"),  
                            base_url=url, 
                        )
                        question = "Describe this image"
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
                                "content":"You are an expert in image description, converting images into corresponding text descriptions"   # system prompt
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
    results.append(get_chat_response_text(args, questions, images))
    return results


def load_train_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    image_folder = "you/image/folder"
    all_response = []

    args = argparse.Namespace(model_name='claude-3.5-sonnet')  

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path):
            question = "Describe this image." 
            response = get_chat_response_batch_text(args, question, image_path)
            print(response)
            all_response.append({
                "image_name": image_name,
                "caption": response
            })

    # save JSON...
    now = datetime.datetime.now()
    formattedtime = now.strftime("%Y%m%d_%H%M%S")
    filename = f"./response/caption.json"
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(all_response, file, ensure_ascii=False, indent=4)
    print(f"saved to {filename}")