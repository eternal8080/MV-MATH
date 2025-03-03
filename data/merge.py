import re
import json
from PIL import Image
import os

input_file_path = r"your/json/file"
output_file_path = r"your/merged/json/file"
output_image_dir = r"your/output/image/dir"
os.makedirs(output_image_dir, exist_ok=True)

# load JSON file
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# resize_images_to_match_height
def resize_images_to_match_height(images, target_height):
    resized_images = []
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        resized_images.append(img.resize((new_width, target_height), Image.LANCZOS))
    return resized_images

base_image_dir = r"your/image/path"  # image root path
for item in data:
    # get image paths
    relative_image_paths = item.get("input_image", [])
    # image_paths = [os.path.join(base_image_dir, os.path.basename(path)) for path in relative_image_paths]
    image_paths = relative_image_paths

    
    if image_paths:
        # load images
        images = []
        for img_path in image_paths:
            img = Image.open(img_path)
            images.append(img)

        # get target height
        target_height = images[0].height
        # resize images to match height
        resized_images = resize_images_to_match_height(images, target_height)

        # merge images
        total_width = sum(img.width for img in resized_images)
        merged_image = Image.new('RGB', (total_width, target_height))
        x_offset = 0
        for img in resized_images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # save merged image
        merged_image_path = os.path.join(output_image_dir, f"{item['problem_id']}_merged.jpg")
        merged_image.save(merged_image_path)
        item["merged_image_path"] = merged_image_path  

        # update image paths
        if "question" in item:
            question_text = re.sub(r'<image_\d+>', '', item["question"])
            item["question"] = question_text + "\n<image_1>"

# save modified JSON
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("done")
