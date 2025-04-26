import json

def filter_json_instances(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: {input_file_path} 不是有效的JSON格式")
        return
    
    # 始终使用列表作为输出格式
    filtered_instances = []
    
    if isinstance(data, list):
        # 如果输入已经是列表，直接过滤
        for instance in data:
            if isinstance(instance, dict) and instance.get('split') == 'val':
                filtered_instances.append(instance)
    elif isinstance(data, dict):
        print("this is a dict================")
        # 对于字典格式，转换为列表
        for key, value in data.items():
            # 检查字典中是否有split字段
            if isinstance(value, dict) and value.get('split') == 'val':
                # 创建一个新的字典，包含原始key和value
                new_instance = {
                    "key": key,  # 保存原始key
                    "data": value  # 保存原始value
                }
                filtered_instances.append(new_instance)
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_instances, f, ensure_ascii=False, indent=2)
        print(f"成功将{len(filtered_instances)}个实例以列表形式写入到 {output_file_path}")
        if filtered_instances:
            print(f"列表中第一个实例结构: {list(filtered_instances[0].keys())}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    import sys
    
    input_file = "/data/user_data/ayliu2/WebQA_train_val.json"
    output_file = "/data/user_data/ayliu2/WebQA_val.json"
    filter_json_instances(input_file, output_file)