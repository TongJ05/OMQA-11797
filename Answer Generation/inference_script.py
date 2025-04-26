import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
from string import Template

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-Next inference script for WebQA")
    parser.add_argument("--model-path", type=str, required=True, help="Path to LLaVA-Next model")
    parser.add_argument("--input-file", type=str, default="/data/user_data/ayliu2/WebQA_val.json", help="Path to input JSON file")
    parser.add_argument("--output-file", type=str, default="/data/user_data/ayliu2/WebQA_results.json", help="Path to output JSON file")
    parser.add_argument("--image-dir", type=str, default="/data/user_data/ayliu2/WebQA_imgs_7z_chunks/images", help="Directory containing images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    return parser.parse_args()

def load_image(image_id, image_dir="/data/user_data/ayliu2/WebQA_imgs_7z_chunks/images"):
    """Load image from local directory using image_id"""
    image_path = os.path.join(image_dir, f"{image_id}.png")
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert('RGB')
        else:
            print(f"Image not found: {image_path}")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def main():
    args = parse_args()
    
    # Import LLaVA-Next here to avoid import errors if dependencies not installed
    try:
        from llava_next.model.builder import load_pretrained_model
        from llava_next.mm_utils import process_images
        from llava_next.conversation import conv_templates
        from llava_next.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    except ImportError:
        raise ImportError("LLaVA-Next dependencies not found. Please install LLaVA-Next.")
    
    print(f"Loading model from {args.model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.device)
    
    print(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    prompt_template = Template("""I'm showing you an image that contains visual evidence to answer a question. Please analyze the image carefully.

Question: ${question}

Please consider:
1. The visual details in the image
2. The title and caption of the image for context
3. Any relevant characteristics of what you observe

Provide a direct, concise answer to the question.""")
    
    results = {}
    
    for item in tqdm(data, desc="Processing"):
        key = item["key"]
        instance = item["data"] if "data" in item else item
        
        # Skip if not validation split
        if instance.get("split") != "val":
            continue
        
        # Get question
        question = instance.get("Q", "").strip('"')
        
        # Initialize variables
        processed_image = None
        image_url = None
        
        # Process only positive image facts for visual QA
        if "img_posFacts" in instance and len(instance["img_posFacts"]) > 0:
            pos_fact = instance["img_posFacts"][0]  # Use the first positive image fact
            image_id = pos_fact.get("image_id")
            
            if image_id:
                image = load_image(image_id, args.image_dir)
                if image:
                    processed_image = process_images([image], image_processor)
                    processed_image = processed_image.to(args.device, dtype=torch.float16)
        
        if processed_image is None:
            print(f"Skipping instance {key} - no valid image found")
            continue
        
        # Prepare prompt
        prompt = prompt_template.substitute(question=question)
        
        # Setup conversation
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        conv_prompt = conv.get_prompt()
        
        # Add image tokens
        if DEFAULT_IMAGE_TOKEN not in conv_prompt:
            # Format with image caption and title if available
            image_context = ""
            if pos_fact.get("title"):
                image_context += f"Title: {pos_fact.get('title')}\n"
            if pos_fact.get("caption"):
                image_context += f"Caption: {pos_fact.get('caption')}\n"
            
            conv_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + image_context + conv_prompt
        
        # Tokenize input
        from llava_next.mm_utils import tokenizer_image_token
        input_ids = tokenizer_image_token(conv_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
        
        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=processed_image,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        
        # Decode output
        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        # Store result with key from JSON
        results[key] = output.strip()
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(results)} instances. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()