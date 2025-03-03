import json
import torch
from PIL import Image
import re
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
from tqdm import tqdm
from llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_PLACEHOLDER
)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

two_shot_examples = """
Question: If a triangle has two sides of length 3 and 4, what is the length of the hypotenuse?  A.10  B.8  C.5  D.4

Answer:  
Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:  
\[c^2 = a^2 + b^2\], where \( a \) and \( b \) are the legs, and \( c \) is the hypotenuse.
Step 2 (Substitute the known values): Given \( a = 3 \) and \( b = 4 \). Substituting these values into the formula:  
\[c^2 = 3^2 + 4^2 = 9 + 16 = 25\]
Step 3 (Calculate the hypotenuse): Taking the square root gives:  
\[c = \sqrt{25} = 5\]
Answer: {C}


Question: In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. Find the area of the right triangle.  A.20  B.10  C.5  D.6
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
Please reason step by step, and put your final answer A, B, C or D within {}. Each step is placed on a new line, using the following format: Step X (Mathematical theorem/basis used): Detailed solution steps. Answer: {A} or Answer: {B} or Answer: {C} or Answer: {D}
"""


def load_image(image_path):
    """Load a single image."""
    image = Image.open(image_path).convert("RGB")
    return image


def load_images(image_paths):
    """Load multiple images."""
    return [load_image(path) for path in image_paths]


def run_inference(model, tokenizer, image_processor, input_images, question, conv_mode, device):
    # Inference for each question and image.
    # Concatenate image tokens
    image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    question = question.replace("<image>", image_token)
    # Load and process images
    images_tensor = process_images(input_images, image_processor, model.config).to(device, dtype=torch.float16)
    
    # Prepare input
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0.2,
            num_beams=1,
            max_new_tokens=2048,
            repetition_penalty=1.2,
            use_cache=True
        )
    
    # Decode output and print answer
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return output


def main(json_file_path, model_path, model_base, answers_file):
    """Main function: read JSON file and infer each question and image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model_name = model_path.split("/")[-1]
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name)
    model.to(device)
    image_folder="/path/to/image_folder"

    # Read JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    with open(answers_file, 'w', encoding='utf-8') as out_f:
        # Iterate over each question and image for inference
        for item in tqdm(data):
            idx = item["problem_id"]
            question = item["translate_question"]
            question_input = re.sub(r'<image_\d+>', '<image>', question)
            question_input = two_shot_examples + question_input
            
            image_paths = item["input_image"]
            image_paths = [os.path.join(image_folder, img).replace("\\", "/") for img in image_paths]
            input_images = load_images(image_paths)
            
            # Determine model conversation mode
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            else:
                conv_mode = "llava_v0"

            # Inference and output results
            answer = run_inference(model, tokenizer, image_processor, input_images, question_input, conv_mode, device)
            # Save output
            result = {
                "problem_id": idx,
                "question": question,
                "model_answer": answer
            }
            json.dump(result, out_f, ensure_ascii=False)
            out_f.write("\n")
            print(f"Problem ID: {item['problem_id']}\nAnswer: {answer}\n")


if __name__ == "__main__":
    # Specify model and JSON file paths
    json_file_path = "/path/to/json_file.json"
    model_path = "/path/to/model"
    answers_file = "/path/to/answers.jsonl"
    
    # Execute main function
    main(json_file_path, model_path, None, answers_file)