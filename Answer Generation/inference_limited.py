import argparse
import json
import os
import torch
import gc
from PIL import Image
from tqdm import tqdm
from string import Template
from transformers import AutoModelForImageTextToText, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA inference script for WebQA with multiple images and texts")
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Path to LLaVA model")
    parser.add_argument("--input-file", type=str, default="/data/user_data/ayliu2/WebQA_val.json", help="Path to input JSON file")
    parser.add_argument("--output-file", type=str, default="/data/user_data/ayliu2/WebQA_results.json", help="Path to output JSON file")
    parser.add_argument("--image-dir", type=str, default="/data/user_data/ayliu2/WebQA_imgs_7z_chunks/images", help="Directory containing images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-instances", type=int, default=10, help="Maximum number of instances to process")
    return parser.parse_args()

def load_images(image_ids, image_dir="/data/user_data/ayliu2/WebQA_imgs_7z_chunks/images"):
    """Load multiple images from local directory using image_ids"""
    images = []
    for image_id in image_ids:
        image_path = os.path.join(image_dir, f"{image_id}.png")
        try:
            if os.path.exists(image_path):
                images.append(Image.open(image_path).convert('RGB'))
            else:
                print(f"Image not found: {image_path}")
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
    return images

def main():
    args = parse_args()
    
    # Set custom Hugging Face cache directory
    import os
    os.environ['HF_HOME'] = '/data/user_data/ayliu2/hugginface'
    os.environ['TRANSFORMERS_CACHE'] = '/data/user_data/ayliu2/hugginface/transformers'
    
    # Configure GPU memory management
    if torch.cuda.is_available() and args.device == "cuda":
        # Print GPU info
        device_props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {device_props.total_memory / 1024**3:.2f} GB")
        
        # Empty cache at the start
        torch.cuda.empty_cache()
        gc.collect()
    
    # Print script configuration
    print("\n=== LLaVA Multi-Image Multi-Text WebQA Inference Script ===")
    print(f"Model path:      {args.model_path}")
    print(f"Input file:      {args.input_file}")
    print(f"Output file:     {args.output_file}")
    print(f"Image directory: {args.image_dir}")
    print(f"Device:          {args.device}")
    print(f"Max instances:   {args.max_instances}")
    print(f"Temperature:     {args.temperature}")
    print(f"Max new tokens:  {args.max_new_tokens}")
    print("========================================\n")
    
    print(f"Loading model from {args.model_path}...")
    
    # Load model and processor using transformers
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        print("Successfully loaded model and processor")
        
        # Print model memory usage
        if torch.cuda.is_available() and args.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"GPU memory reserved: {memory_reserved:.2f} GB")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise Exception(f"Could not load model: {e}")

    print(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    prompt_template = Template("""I'm showing you visual and textual evidence to answer a question. Please analyze the images and texts carefully.

Question: ${question}

Visual Context Images:
${image_details}

Textual Context:
${text_details}

Please consider:
1. The visual details in the images
2. The titles and captions of the images for context
3. The provided textual information
4. Any relevant characteristics of what you observe

Provide a direct, concise answer to the question.""")
    
    results = {}
    
    # Limit to first args.max_instances valid instances
    processed_count = 0
    valid_instances = []
    
    print("Scanning for valid instances...")
    scan_progress = tqdm(data, desc="Scanning instances")
    instances_checked = 0
    instances_skipped = 0
    
    for item in scan_progress:
        instances_checked += 1
        key = item["key"]
        instance = item["data"] if "data" in item else item
        
        # Update progress bar description
        scan_progress.set_description(
            f"Scanning instances | Found: {len(valid_instances)} | Checked: {instances_checked} | Skipped: {instances_skipped}"
        )
        
        # Skip if not validation split
        if instance.get("split") != "val":
            instances_skipped += 1
            continue
            
        # Skip if no positive image or text facts
        if not "img_posFacts" in instance or len(instance["img_posFacts"]) == 0 or \
           not "txt_posFacts" in instance or len(instance["txt_posFacts"]) == 0:
            instances_skipped += 1
            continue
            
        # Check if images can be loaded
        image_ids = [pos_fact.get("image_id") for pos_fact in instance["img_posFacts"]]
        if not all(image_ids):
            instances_skipped += 1
            continue
            
        # Try to load images to verify they exist
        images = load_images(image_ids, args.image_dir)
        if len(images) != len(image_ids):
            instances_skipped += 1
            continue
            
        # Add to valid instances
        valid_instances.append((key, instance))
        
        # Break if we've reached our limit
        if len(valid_instances) >= args.max_instances:
            break
    
    print(f"Found {len(valid_instances)} valid instances out of {instances_checked} checked ({instances_skipped} skipped).")
    
    processing_progress = tqdm(valid_instances, desc="Processing instances")
    instance_num = 0
    
    for key, instance in processing_progress:
        instance_num += 1
        # Get question
        question = instance.get("Q", "").strip('"')
        
        # Prepare image details
        image_details = []
        images_to_process = []
        for img_pos_fact in instance["img_posFacts"]:
            image_id = img_pos_fact.get("image_id")
            image = load_images([image_id], args.image_dir)[0]
            images_to_process.append(image)
            
            # Build image detail string
            img_detail = f"Image {len(image_details) + 1}:"
            if img_pos_fact.get("title"):
                img_detail += f" Title: {img_pos_fact.get('title')}"
            if img_pos_fact.get("caption"):
                img_detail += f" Caption: {img_pos_fact.get('caption')}"
            image_details.append(img_detail)
        
        # Prepare text details
        text_details = []
        for txt_pos_fact in instance["txt_posFacts"]:
            text_detail = txt_pos_fact.get("text", "")
            if text_detail:
                text_details.append(text_detail)
        
        # Update progress description
        processing_progress.set_description(
            f"Processing instance {instance_num}/{len(valid_instances)} | Key: {key[:8]}..."
        )
        
        # Prepare prompt
        prompt = prompt_template.substitute(
            question=question, 
            image_details="\n".join(image_details), 
            text_details="\n".join(text_details)
        )
        
        # Clean up memory before processing
        if torch.cuda.is_available() and args.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Use the processor to prepare inputs
            inputs = processor(
                text=prompt,
                images=images_to_process,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            processing_progress.set_description(
                f"Processing instance {instance_num}/{len(valid_instances)} | Key: {key[:8]}... | Generating response"
            )
            
            with torch.inference_mode():
                # Prepare generation parameters
                generation_kwargs = {
                    'max_new_tokens': args.max_new_tokens,
                }
                
                # Set sampling parameters only if temperature > 0
                if args.temperature > 0:
                    generation_kwargs.update({
                        'do_sample': True,
                        'temperature': args.temperature,
                        'top_p': 0.6,
                    })
                else:
                    generation_kwargs.update({
                        'do_sample': False
                    })
                    
                # Generate response
                output_ids = model.generate(**inputs, **generation_kwargs)
            
            # Decode output
            output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nWARNING: Out of memory for instance {instance_num}. Processing images individually...")
                
                # Clean up memory
                if torch.cuda.is_available() and args.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Process one image at a time and collect descriptions
                image_descriptions = []
                
                for idx, image in enumerate(images_to_process):
                    try:
                        # Process single image
                        single_input = processor(
                            text=f"Describe this image briefly:",
                            images=[image],
                            return_tensors="pt"
                        ).to(model.device)
                        
                        with torch.inference_mode():
                            single_output_ids = model.generate(
                                **single_input,
                                max_new_tokens=100,
                                do_sample=False
                            )
                        
                        # Get description
                        image_desc = processor.batch_decode(single_output_ids, skip_special_tokens=True)[0]
                        image_descriptions.append(f"Image {idx+1} description: {image_desc}")
                        
                        # Clean up memory
                        if torch.cuda.is_available() and args.device == "cuda":
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                    except Exception as inner_e:
                        print(f"Error processing image {idx+1}: {inner_e}")
                        image_descriptions.append(f"Image {idx+1}: {image_details[idx]}")
                
                # Create a prompt with image descriptions instead of actual images
                text_only_prompt = f"""Question: {question}

Visual Context (from descriptions):
{' '.join(image_descriptions)}

Textual Context:
{' '.join(text_details[:1]) if text_details else 'No textual context available.'}

Answer the question directly and concisely."""
                
                try:
                    # Process text-only input
                    text_inputs = processor(
                        text=text_only_prompt,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    with torch.inference_mode():
                        text_output_ids = model.generate(
                            **text_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False
                        )
                    
                    output = processor.batch_decode(text_output_ids, skip_special_tokens=True)[0]
                    
                except Exception as fallback_e:
                    print(f"Fallback also failed: {fallback_e}")
                    output = f"Error: Could not process this instance due to memory constraints."
            else:
                print(f"\nError processing instance {instance_num}: {e}")
                output = f"Error: {str(e)}"
        
        # Store result with key from JSON
        results[key] = output.strip()
        processed_count += 1
        
        # Print memory usage
        if torch.cuda.is_available() and args.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Save results after each instance is processed
        print(f"Saving intermediate results after processing instance {instance_num}/{len(valid_instances)}...")
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output_file}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total instances processed: {len(results)}")
    print(f"Success rate: {len(results)}/{len(valid_instances)} ({(len(results)/len(valid_instances))*100:.1f}%)")
    print(f"Results saved to: {args.output_file}")
    print(f"First few keys: {list(results.keys())[:3]}")
    print("===========================")

if __name__ == "__main__":
    main()