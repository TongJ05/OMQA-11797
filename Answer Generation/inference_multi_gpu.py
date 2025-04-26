import argparse
import json
import os
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime
from string import Template
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-Next Multi-GPU Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to LLaVA-Next model")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--max-instances", type=int, default=0, help="Maximum number of instances to process (0 = all)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate")
    return parser.parse_args()

def load_image(image_id, image_dir):
    """从本地目录使用image_id加载图像"""
    image_path = os.path.join(image_dir, f"{image_id}.png")
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert('RGB')
        else:
            print(f"找不到图像: {image_path}")
            return None
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return None

def find_valid_instances(input_file, image_dir, max_instances=0):
    """找出所有有效的实例"""
    print(f"从 {input_file} 加载数据")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    valid_instances = []
    
    print("扫描有效实例...")
    scan_progress = tqdm(data, desc="扫描实例")
    instances_checked = 0
    instances_skipped = 0
    
    for item in scan_progress:
        instances_checked += 1
        key = item["key"]
        instance = item["data"] if "data" in item else item
        
        # 更新进度条描述
        scan_progress.set_description(
            f"扫描实例 | 已找到: {len(valid_instances)} | 已检查: {instances_checked} | 已跳过: {instances_skipped}"
        )
        
        # 跳过非验证集数据
        if instance.get("split") != "val":
            instances_skipped += 1
            continue
            
        # 跳过无正面图像事实的数据
        if not "img_posFacts" in instance or len(instance["img_posFacts"]) == 0:
            instances_skipped += 1
            continue
            
        # 检查图像是否可以加载
        pos_fact = instance["img_posFacts"][0]
        image_id = pos_fact.get("image_id")
        if not image_id:
            instances_skipped += 1
            continue
            
        # 尝试加载图像验证其存在
        image = load_image(image_id, image_dir)
        if image is None:
            instances_skipped += 1
            continue
            
        # 添加到有效实例
        valid_instances.append((key, instance, pos_fact))
        
        # 达到限制时停止
        if max_instances > 0 and len(valid_instances) >= max_instances:
            break
    
    print(f"找到 {len(valid_instances)} 个有效实例，共检查 {instances_checked} 个 (跳过 {instances_skipped} 个).")
    return valid_instances
    
def process_batch(rank, args, instance_batches, result_queue):
    """在特定GPU上处理一批实例"""
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    # 导入LLaVA-Next
    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    except ImportError:
        raise ImportError("未找到LLaVA-Next依赖，请安装LLaVA-Next。")
    
    print(f"[GPU {rank}] 从 Hugging Face 加载模型")
    
    # 直接从 Hugging Face 下载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path="lmms-lab/llava-next-qwen-32b",  # 直接使用 Hugging Face 模型标识符
        model_base="Qwen/Qwen-VL",      # Qwen基础模型
        model_name="qwen",              # 使用 "qwen" 作为模型名称
        device_map="auto",              # 设备映射
        torch_dtype="float16",           # 精度
         trust_remote_code=True 
    )
    
    prompt_template = Template("""You are an expert visual analyst. I'm providing you with an image to help answer a specific question. 

Image Context:
${image_context}

Question: ${question}

Please analyze the image carefully and provide a precise, concise answer that directly addresses the question. Consider:
1. Specific visual details in the image
2. Relevant context from the image title or caption
3. Any key features that help answer the question

Your response should be:
- Directly related to the question
- Based solely on the visual information provided
- Clear and to the point""")
    
    batch = instance_batches[rank]
    batch_size = len(batch)
    
    local_results = {}
    processing_progress = tqdm(
        batch, 
        desc=f"[GPU {rank}] Processing Instances", 
        position=rank
    )
    
    for i, (key, instance, pos_fact) in enumerate(processing_progress):
        # Get question
        question = instance.get("Q", "").strip('"')
        image_id = pos_fact.get("image_id")
        
        # Update progress description
        processing_progress.set_description(
            f"[GPU {rank}] Processing Instance {i+1}/{batch_size} | Key: {key[:8]}..."
        )
        
        # Load image
        image = load_image(image_id, args.image_dir)
        if image:
            processed_image = process_images([image], image_processor)
            processed_image = processed_image.to(device, dtype=torch.float16)
        else:
            print(f"[GPU {rank}] Skipping instance {key} - Unable to load image")
            continue
        
        # Prepare image context
        image_context = ""
        if pos_fact.get("title"):
            image_context += f"Image Title: {pos_fact.get('title')}\n"
        if pos_fact.get("caption"):
            image_context += f"Image Caption: {pos_fact.get('caption')}\n"
        
        # Prepare prompt
        prompt = prompt_template.substitute(
            image_context=image_context,
            question=question
        )
        
        # Set up conversation
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        conv_prompt = conv.get_prompt()
        
        # Add image tokens
        if DEFAULT_IMAGE_TOKEN not in conv_prompt:
            conv_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + conv_prompt
        
        # Tokenize input
        input_ids = tokenizer_image_token(conv_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        
        # Generate response
        processing_progress.set_description(
            f"[GPU {rank}] Processing Instance {i+1}/{batch_size} | Key: {key[:8]}... | Generating Response"
        )
        
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
        
        # Save results
        local_results[key] = output.strip()
        
        # Put result in queue
        result_queue.put((key, output.strip()))
    
    # Mark GPU processing complete
    result_queue.put((None, None))
    print(f"[GPU {rank}] Completed processing {len(local_results)} instances")

# The rest of the script remains the same

def main():
    args = parse_args()
    
    # 打印脚本配置
    print("\n=== LLaVA-Next WebQA多GPU推理脚本 ===")
    print(f"模型路径:       {args.model_path}")
    print(f"输入文件:       {args.input_file}")
    print(f"输出文件:       {args.output_file}")
    print(f"图像目录:       {args.image_dir}")
    print(f"GPU数量:        {args.num_gpus}")
    print(f"最大实例数:     {args.max_instances}")
    print(f"温度:           {args.temperature}")
    print(f"最大新token数:  {args.max_new_tokens}")
    print("=========================================\n")
    
    # 检查可用GPU数量
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.num_gpus:
        print(f"警告: 请求了 {args.num_gpus} 个GPU，但只有 {available_gpus} 个可用. 将使用所有可用GPU.")
        args.num_gpus = available_gpus
    
    # 找出所有有效实例
    valid_instances = find_valid_instances(args.input_file, args.image_dir, args.max_instances)
    
    if not valid_instances:
        print("未找到有效实例，退出。")
        return
    
    # 将实例分成多个批次，每个GPU一个批次
    num_instances = len(valid_instances)
    batch_size = math.ceil(num_instances / args.num_gpus)
    instance_batches = []
    
    for i in range(args.num_gpus):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_instances)
        instance_batches.append(valid_instances[start_idx:end_idx])
    
    print(f"将 {num_instances} 个实例分成 {args.num_gpus} 个批次，每个批次约 {batch_size} 个实例")
    
    # 使用多处理
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    
    # 启动每个GPU的处理
    for rank in range(args.num_gpus):
        p = mp.Process(target=process_batch, args=(rank, args, instance_batches, result_queue))
        p.start()
        processes.append(p)
    
    # 收集结果
    results = {}
    completed_gpus = 0
    
    start_time = time.time()
    print(f"开始多GPU处理，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 进度条
    progress_bar = tqdm(total=num_instances, desc="总进度")
    processed_count = 0
    
    while completed_gpus < args.num_gpus:
        key, result = result_queue.get()
        if key is None:  # 一个GPU完成
            completed_gpus += 1
        else:  # 实例结果
            results[key] = result
            processed_count += 1
            progress_bar.update(1)
            
            # 显示处理速度
            elapsed = time.time() - start_time
            if processed_count > 0 and elapsed > 0:
                instances_per_sec = processed_count / elapsed
                remaining = (num_instances - processed_count) / instances_per_sec if instances_per_sec > 0 else 0
                progress_bar.set_postfix({
                    "速度": f"{instances_per_sec:.2f} 实例/秒", 
                    "剩余": f"{remaining/60:.1f} 分钟"
                })
    
    progress_bar.close()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 保存结果
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    total_time = time.time() - start_time
    print("\n=== 处理摘要 ===")
    print(f"总处理实例数: {len(results)}")
    print(f"成功率: {len(results)}/{num_instances} ({(len(results)/num_instances)*100:.1f}%)")
    print(f"总处理时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"平均速度: {num_instances/total_time:.2f} 实例/秒")
    print(f"结果保存到: {args.output_file}")
    if results:
        print(f"前几个键: {list(results.keys())[:3]}")
    print("===================")

if __name__ == "__main__":
    main()