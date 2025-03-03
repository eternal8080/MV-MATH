import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math


system_message = "You are an assistant for solving math problems. Your input consists of a math question and images. Your task is to output the solution steps and the answer. The output format should be a step-by-step approach. Each question is multiple choice with one correct answer. Your final answer must be one of A, B, C, or D, For example: {A},{B},{C} or {D}"
# Two-shot conversation examples
two_shot_conversations = [
    {"from": "human", "value": "If a triangle has two sides of length 3 and 4, what is the length of the hypotenuse?  A.10  B.8  C.5  D.4"},
    {"from": "gpt", "value": """
Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:
\[c^2 = a^2 + b^2\], where \(a\) and \(b\) are the legs, and \(c\) is the hypotenuse.
Step 2 (Substitute the known values): Given \(a = 3\) and \(b = 4\). Substituting these values into the formula:
\[c^2 = 3^2 + 4^2 = 9 + 16 = 25\]
Step 3 (Calculate the hypotenuse): Taking the square root gives:
\[c = \sqrt{25} = 5\]
Answer: {C}
"""},

    {"from": "human", "value": "In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. Find the area of the right triangle.  A.20  B.10  C.5  D.6"},
    {"from": "gpt", "value": """
Step 1 (Mathematical theorem used: Pythagorean theorem): We first use the Pythagorean theorem to find the length of \(BC\). The formula is:
\[AC^2 = AB^2 + BC^2\], where \(AC\) is the hypotenuse, and \(AB\) and \(BC\) are the legs.
Step 2 (Substitute the known values): Given \(AC = 5\) and \(AB = 4\). Substituting these values:
\[5^2 = 4^2 + BC^2 \implies 25 = 16 + BC^2\]
Step 3 (Solve for \(BC\)):
\[BC^2 = 25 - 16 = 9 \implies BC = \sqrt{9} = 3\]
Step 4 (Calculate the area): The area of the right triangle is given by \( \frac{1}{2} \times AB \times BC \). Substituting the known values:
\[\text{Area} = \frac{1}{2} \times 4 \times 3 = 6\]
Answer: {D}
"""}
]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = system_message) -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_path)
    print(model_name)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map="auto")

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    image_folder = args.image_folder

    with open(answers_file, 'w', encoding='utf-8') as out_f:
        for line in tqdm(questions, desc="Processing questions", unit="sample"):
            idx = line["sample_id"]

            image_files = [os.path.join(image_folder, img).replace("\\", "/") for img in line["image"]]
            qs = line["conversations"][0]["value"]
            cur_prompt = args.extra_prompt + qs

            args.conv_mode = "qwen_1_5"

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Get user's current question
            user_conversation = line["conversations"][0]  # User's input question
            model_placeholder = {'from': 'gpt', 'value': None}  # Model placeholder response

            # Combine two-shot example with user question and model placeholder into a complete conversation sequence
            all_conversations = two_shot_conversations + [user_conversation, model_placeholder]
            # Pass to preprocess_qwen function
            input_ids = preprocess_qwen(
                all_conversations,  # Conversation list in the expected format
                tokenizer=tokenizer,
                has_image=True  # If images need to be processed, set to True
            ).cuda()
            img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

            image_tensors = []
            target_device = 'cuda'
            for image_file in image_files:
                image = Image.open(os.path.join(image_folder, image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(target_device)
                image_tensors.append(image_tensor.half().to(target_device))
            # image_tensors = torch.cat(image_tensors, dim=0)
            
            input_ids = input_ids.to(target_device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True if args.temperature > 0 else False,
                    # do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)
            # Decode output and print answer
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # Save output
            result = {
                "problem_id": idx,
                "question": qs,
                "model_answer": outputs
            }
            json.dump(result, out_f, ensure_ascii=False)
            out_f.write("\n")

            print(idx, outputs)
        print(f"Results saved to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/path/to/model")   # suitable for llava-next and llava-onevision
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/path/to/images")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="/path/to/questions.json")
    parser.add_argument("--answers-file", type=str, default="/path/to/answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=100000)
    args = parser.parse_args()

    eval_model(args)