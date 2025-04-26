import json
import argparse
import os
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_images(image_ids, image_folder):
    """
    Load images based on image IDs from the specified folder.
    Similar to the example code's load_images_from_entry function.
    
    Args:
        image_ids: List of image IDs
        image_folder: Path to the folder containing images
        
    Returns:
        List of loaded PIL images
    """
    images = []
    missing_images = []
    
    for image_id in image_ids:
        # Try different file extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            image_path = os.path.join(image_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    images.append(img)
                    break
                except Exception as e:
                    logger.warning(f"Error loading image {image_id}: {str(e)}")
                    missing_images.append(image_id)
        else:  # No break occurred, so no file was found
            missing_images.append(image_id)
    
    if missing_images:
        logger.warning(f"Could not find or load images for IDs: {missing_images}")
    
    return images

def save_results(results, output_file):
    """
    Save results to output file atomically to prevent data loss if interrupted.
    
    Args:
        results: Dictionary with results
        output_file: Path to output file
    """
    # Write to a temporary file first, then rename to ensure atomic operation
    temp_file = f"{output_file}.tmp"
    with open(temp_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Rename temp file to final output file (atomic operation on most systems)
    os.replace(temp_file, output_file)

def process_prompts(prompts_data, model, processor, image_folder, output_file):
    """
    Process all prompts using the LLaVA model and save results.
    Using the conversation approach from the example code.
    
    Args:
        prompts_data: Dictionary with instance IDs and prompts
        model: LLaVA model
        processor: LLaVA processor
        image_folder: Folder containing images
        output_file: Path to output JSON file
    """
    # Initialize results - load existing results if available
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} existing results from {output_file}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing results file, starting fresh")
            results = {}
    else:
        results = {}
    
    # Get list of instance IDs to process (filter out already processed ones)
    to_process = [id for id in prompts_data if id not in results]
    logger.info(f"Processing {len(to_process)} instances out of {len(prompts_data)} total")
    
    # Process each prompt
    for instance_id in tqdm(to_process, desc="Processing prompts"):
        data = prompts_data[instance_id]
        image_ids = data["image_ids"]
        prompt_text = data["prompt"]
        
        # Extract the question from the prompt
        question_start = prompt_text.find('Question: "')
        question_end = prompt_text.find('"', question_start + 11) if question_start != -1 else -1
        if question_start != -1 and question_end != -1:
            question = prompt_text[question_start + 11:question_end]
        else:
            # Fallback to using the whole prompt
            question = prompt_text
        
        try:
            # Load images
            images = load_images(image_ids, image_folder)
            
            if not images:
                logger.warning(f"No images found for instance {instance_id}, skipping")
                results[instance_id] = {"answer": "ERROR: No images found", "image_ids": image_ids}
                
                # Save results immediately
                save_results(results, output_file)
                continue
            
            # Prepare conversation template similar to the example code
            conversation = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"} for _ in images] +
                        [{"type": "text", "text": prompt_text}]
                    )
                }
            ]
            
            # Apply chat template
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process with model
            inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
            
            # Generate answer
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            answer = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            # Store result
            results[instance_id] = {
                "answer": answer,
                "image_ids": image_ids,
                "question": question
            }
            
            # Save results immediately after each processing
            save_results(results, output_file)
            logger.info(f"Processed and saved instance {instance_id}")
                
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error processing instance {instance_id}: {str(e)}\n{error_detail}")
            results[instance_id] = {
                "answer": f"ERROR: {str(e)}", 
                "image_ids": image_ids,
                "question": question if 'question' in locals() else prompt_text,
                "error_detail": error_detail
            }
            
            # Save results even after error
            save_results(results, output_file)
    
    logger.info(f"Processing complete. Final results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process prompts using LLaVA model")
    parser.add_argument("--input", "-i", type=str, required=True, 
                      help="Path to JSON file with prompts")
    parser.add_argument("--output", "-o", type=str, required=True,
                      help="Path to output JSON file to store the results")
    parser.add_argument("--image_folder", "-f", type=str, default="../images",
                      help="Path to folder containing images")
    parser.add_argument("--model", "-m", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
                      help="LLaVA model to use")
    
    args = parser.parse_args()
    
    # Load prompts
    with open(args.input, 'r') as f:
        prompts_data = json.load(f)
    
    logger.info(f"Loaded {len(prompts_data)} prompts from {args.input}")
    
    # Initialize model and processor
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    
    # Process prompts
    process_prompts(prompts_data, model, processor, args.image_folder, args.output)

if __name__ == "__main__":
    main()