import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from tqdm import tqdm  # Import progress bar library
from datetime import datetime  # Import datetime module

# Load model and processor
cache_dir = "/path/to/cache"  # Replace with your desired path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
    cache_dir=cache_dir
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir=cache_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read JSONL file and calculate total number of questions
json_file = "/path/to/json_file.json"
image_folder = "/path/to/image_folder"

# Get current time and format it
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"/path/to/output_file.jsonl"

# Check if the output file directory exists, create it if not
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Two-shot reasoning examples
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
Step 2 (Substitute the known values): Given \( AC =5 \) and \( AB = 4 \). Substituting these values:  
\[5^2 = 4^2 + BC^2 \implies 25 = 16 + BC^2\]
Step 3 (Solve for \( BC \)):  
\[BC^2 = 25 - 16 = 9 \implies BC = \sqrt{9} = 3\]
Step 4 (Calculate the area): The area of the right triangle is given by \( \frac{1}{2} \times AB \times BC \). Substituting the known values:  
\[ \text{Area} = \frac{1}{2} \times 4 \times 3 = 6\]
Answer: {D}

"""


# Process JSON file
with open(json_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
    data = json.load(f)  # Parse the entire JSON file, not line by line

    for problem in tqdm(data, total=len(data), desc="Processing"):  # Use len(data) to calculate total number of questions
        question_id = problem['problem_id']
        question_text = problem['translate_question']
        image_paths = [os.path.join(image_folder, img).replace("\\", "/") for img in problem['_input_image']]
        
        # Generate three-shot reasoning message template
        prompt = f"{two_shot_examples}\nQuestion: {question_text}"
    
        # Construct messages containing multi-image input and question
        messages = [
            {
                "role": "system", 
                "content": "You are an assistant for solving math problems. Your input consists of a math question and images. Your task is to output the solution steps and the answer. The output format should be a step-by-step approach. Each question is multiple choice with one correct answer. Your final answer must be one of A, B, C, or D, For example: {A},{B},{C} or {D}"

            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path} for img_path in image_paths
                ] + [{"type": "text", "text": prompt}]
            }
        ]

        # Prepare inference input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.2, repetition_penalty=1.2)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Save output
        result = {
            "problem_id": question_id,
            "question": question_text,
            "model_answer": output_text[0] if output_text else None
        }
        json.dump(result, out_f, ensure_ascii=False)
        out_f.write("\n")

        print(question_id, output_text[0])
    print(f"Results saved to {output_file}")