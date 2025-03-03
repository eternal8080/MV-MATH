import json
import torch
from transformers import AutoModelForCausalLM
import re 
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
import datetime
import os
from tqdm import tqdm 

# specify the path to the model and cache directory
model_path = "deepseek-ai/deepseek-vl-7b-chat"
cache_dir = "/path/to/cache"  

# load model
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir=cache_dir)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, cache_dir=cache_dir
)
vl_gpt = vl_gpt.to("cuda:0", torch.bfloat16).eval()

# load json file
json_path = "/path/to/json_file.json"
with open(json_path, 'r') as f:
    data = json.load(f)
image_folder = "/path/to/image_folder"

# Two-shot reasoning examples
two_shot_examples = """
Question: If a triangle has two sides of length 2 and 3, what is the length of the hypotenuse?  A.5  B.4  C.3  D.2

Answer:  
Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:  
\[c^2 = a^2 + b^2\], where \( a \) and \( b \) are the legs, and \( c \) is the hypotenuse.
Step 2 (Substitute the known values): Given \( a = 2 \) and \( b = 3 \). Substituting these values into the formula:  
\[c^2 = 2^2 + 3^2 = 4 + 9 = 13\]
Step 3 (Calculate the hypotenuse): Taking the square root gives:  
\[c = \sqrt{13}\]
Answer: {C}


Question: In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. Find the area of the right triangle.  A.10  B.8  C.6  D.3
Answer:  
Step 1 (Mathematical theorem used: Pythagorean theorem): We first use the Pythagorean theorem to find the length of \( BC \). The formula is:  
\[AC^2 = AB^2 + BC^2\], where \( AC \) is the hypotenuse, and \( AB \) and \( BC \) are the legs.
Step 2 (Substitute the known values): Given \( AC = 5 \) and \( AB = 4 \). Substituting these values:  
\[5^2 = 4^2 + BC^2 \implies 25 = 16 + BC^2\]
Step 3 (Solve for \( BC \)):  
\[BC^2 = 25 - 16 = 9 \implies BC = \sqrt{9} = 3\]
Step 4 (Calculate the area): The area of the right triangle is given by \( \frac{1}{2} \times AB \times BC \). Substituting the known values:  
\[\text{Area} = \frac{1}{2} \times 4 \times 3 = 6\]
Answer: {C}

"""

# Get current time for result file naming
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_file_path = f"/path/to/result_file_{time_str}.jsonl"

# Read completed sample IDs
completed_ids = set()
try:
    with open(result_file_path, 'r') as result_file:
        for line in result_file:
            result = json.loads(line)
            completed_ids.add(result["problem_id"])
except FileNotFoundError:
    print("Result file not found. Starting fresh.")

# Reason for each sample's question and save results
with open(result_file_path, 'a') as result_file:
    for item in tqdm(data, desc="Processing questions", unit="sample"):
        if item["problem_id"] in completed_ids:
            continue  # Skip completed samples
        
        question_text = re.sub(r'<image_\d+>', '<image_placeholder>', item["translate_question"])
        prompt = f"{two_shot_examples}\nQuestion: {question_text} please reason step by step."
        image_paths = [os.path.join(image_folder, img).replace("\\", "/") for img in item["input_image"]]

        # Extract questions and image paths from json
        conversation = [
            {
                "role": "system",
                "content": "You are an assistant for solving math problems. Your input consists of a math question and images. Your task is to output the solution steps and the answer. The output format should be a step-by-step approach. Each question is multiple choice with one correct answer. Your final answer must be one of A, B, C, or D"
            },
            {
                "role": "User",
                "content": prompt,
                "images": image_paths,
            },
            {"role": "Assistant", "content": ""},
        ]

        # Load images and prepare input
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # Run image encoder to get image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Run model to get response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=2048,
            temperature=0.2,
            repetition_penalty=1.2,
            do_sample=False,
            use_cache=True,
        )

        # Decode output and print answer
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # Save result to jsonl file
        result = {
            "problem_id": item["problem_id"],
            "question_text": question_text,
            "model_answer": answer
        }
        result_file.write(json.dumps(result) + "\n")
        print(f"Problem ID: {item['problem_id']}", question_text, answer)