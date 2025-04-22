import json
import collections
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LongformerTokenizer, LongformerForSequenceClassification, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
from torch.amp import autocast, GradScaler
import random
from collections import Counter
import math


# classes: text1, text2, text3, text4, text5, img1, img2, img3
# meaning: text1 = text question, require 1 context; img2 = image question, require 2 contexts, ...
# labels: [0, 1, 2, 3, 4, 5, 6, 7]
class QTypeDatasetWithContext(Dataset):
    def __init__(self, data):
        self.data = [
            {
                "Q": item["Q"],  # question text
                "split": item["split"],  # train/val
                "q_type": item["q_type"],  # img/txt
                "difficulty": item["hop_estimate"],  # 1 img_posFact = 2, 1 txt_posFact = 1
                "context": item["context"],  # context string
                "type_with_difficulty": item["type_with_difficulty"]  # final label, see above for mapping
            }
            for item in data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def prepare_data_train_val(json_path, debugging_partition=False):
    # set debugging_partition to True to select 1000 samples
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_val = [value for key, value in data.items()]
    if debugging_partition:
        train_val = random.sample(train_val, 1000)

    for q in train_val:
        q['q_type'] = 'img' if q['img_posFacts'] else 'txt'
        q['hop_estimate'] = 2 * len(q['img_posFacts']) + len(q['txt_posFacts'])

        context_str = ''
        context_count = 1
        q['type_with_difficulty'] = None
        if q['q_type'] == 'img':  # image question: no txt_posFacts
            # construct context
            context_str_candidates = []
            for i in q['img_posFacts']:
                context_str_candidates.append(f"(image):\nTitle:{i['title']}\nCaption:{i['caption']}\n")
            for i in q['img_negFacts']:
                context_str_candidates.append(f"(image):\nTitle:{i['title']}\nCaption:{i['caption']}\n")
            for i in q['txt_negFacts']:
                context_str_candidates.append(f"(text):\nTitle:{i['title']}\nContent:{i['fact']}\n")
            random.shuffle(context_str_candidates)
            for context in context_str_candidates:
                context_beginning = f"Context {context_count} "
                context_count += 1
                context_str += (context_beginning + context)

            # set detailed class for image questions
            if len(q['img_posFacts']) == 1:
                q['type_with_difficulty'] = 5
            elif len(q['img_posFacts']) == 2:
                q['type_with_difficulty'] = 6
            else:
                q['type_with_difficulty'] = 7

        else:  # text question: no img_posFacts
            context_str_candidates = []
            for i in q['img_negFacts']:
                context_str_candidates.append(f"(image):\nTitle:{i['title']}\nCaption:{i['caption']}\n")
            for i in q['txt_negFacts']:
                context_str_candidates.append(f"(text):\nTitle:{i['title']}\nContent:{i['fact']}\n")
            for i in q['txt_posFacts']:
                context_str_candidates.append(f"(text):\nTitle:{i['title']}\nContent:{i['fact']}\n")

            random.shuffle(context_str_candidates)
            for context in context_str_candidates:
                context_beginning = f"Context {context_count} "
                context_count += 1
                context_str += (context_beginning + context)

            # set detailed class for text questions
            if len(q['txt_posFacts']) == 1:
                q['type_with_difficulty'] = 0
            elif len(q['txt_posFacts']) == 2:
                q['type_with_difficulty'] = 1
            elif len(q['txt_posFacts']) == 3:
                q['type_with_difficulty'] = 2
            elif len(q['txt_posFacts']) == 4:
                q['type_with_difficulty'] = 3
            else:
                q['type_with_difficulty'] = 4
        q['context'] = context_str

    q_types = [item.get("q_type", "Unknown") for item in train_val]
    hop_estimates = [item.get("hop_estimate", None) for item in train_val]

    # print("Difficulty distribution in val set:")
    val_difficulty = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for item in train_val:
        if item["split"] == "val":
           val_difficulty[item['hop_estimate']] += 1
    # print(val_difficulty)

    # Count occurrences
    q_type_counts = collections.Counter(q_types)
    hop_estimates_counts = collections.Counter(hop_estimates)

    # Show statistics
    # print("\nQ_Type Distribution:")
    # print(q_type_counts)
    #
    # print("\nHop_Estimate Statistics:")
    # print(hop_estimates_counts)
    data = [item for item in train_val]
    dataset = QTypeDatasetWithContext(data)

    return dataset


class DetailedTypeDatasetWithContext(Dataset):
    # contain tokenized text and label
    def __init__(self, data, tokenizer, split='train', max_length=4096):  # Set max_length for Longformer
        self.data = [item for item in data if item['split'] == split]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['Q'] + '\n' + item['context']
        # question = item['Q'] # no context option

        inputs = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        label = torch.tensor(item['type_with_difficulty'], dtype=torch.long)

        return input_ids, attention_mask, label


def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)  # Ensure classification labels are LongTensor

    return input_ids, attention_mask, labels


if __name__ == "__main__":
    batch_size = 4
    train_epoch = 3
    initial_lr = 1e-5
    min_lr = initial_lr * 0.01
    save_path = "saved_model_detailed_classification_longformer_qonly"
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    data = prepare_data_train_val('../WebQA_train_val.json')

    train_labels = [item['type_with_difficulty'] for item in data if item['split'] == 'train']

    # Count label frequencies
    label_counts = Counter(train_labels)
    print("Class distribution:", label_counts)

    num_classes = 8
    total_samples = sum(label_counts.values())

    raw_weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Avoid division by zero
        raw_weight = total_samples / (num_classes * count)
        capped_weight = min(raw_weight, 400)  # Cap the weight at 400
        flattened_weight = math.log(capped_weight + 1)  # Log flatten
        raw_weights.append(flattened_weight)

    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
    for i, w in enumerate(class_weights):
        print(f"Class {i} weight: {w:.4f}")

    # print(data[-1]['context'])
    # time.sleep(200)


    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    train_dataset = DetailedTypeDatasetWithContext(data, tokenizer, split='train')
    val_dataset = DetailedTypeDatasetWithContext(data, tokenizer, split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Initialize model for classification
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=8)
    model.to(device)

    # Optimizer & lr Scheduler
    optimizer = AdamW(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=min_lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(device='cuda')

    # train + val
    for epoch in range(train_epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        batch_count = 0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_epoch} Training", leave=True)

        for input_ids, attention_mask, labels in train_iterator:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training using autocast
            with autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            # Compute batch accuracy
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_iterator.set_postfix(loss=total_loss / len(train_loader), acc=correct / total,
                                       lr=optimizer.param_groups[0]['lr'])

            batch_count += 1  # Increment batch count, debugging

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"\nEpoch {epoch + 1} | Training Loss: {avg_train_loss:.4f} | Training Accuracy: {train_accuracy:.4f}")
        scheduler.step()

        # val
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        batch_count = 0

        all_preds = []
        all_labels = []

        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{train_epoch} Validation", leave=True)

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_iterator:

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                with autocast(device_type='cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)  # Get predicted class (0-7)
                preds = preds.cpu().numpy()
                true_labels = labels.cpu().numpy()

                # Update overall accuracy
                val_correct += (preds == true_labels).sum()
                val_total += labels.size(0)
                all_preds.extend(preds)
                all_labels.extend(true_labels)

                val_iterator.set_postfix(loss=val_loss / len(val_loader))

                batch_count += 1  # Increment batch count, debugging

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        print(f"\nEpoch {epoch + 1} | Validation Loss: {avg_val_loss:.4f} | Overall Accuracy: {val_accuracy:.4f}")

        # Save the model
        model_save_path = os.path.join(save_path, f"longformer_difficulty_classification_{epoch + 1}.pth")
        tokenizer_save_path = os.path.join(save_path, f"tokenizer_difficulty_{epoch + 1}")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save tokenizer
        tokenizer.save_pretrained(tokenizer_save_path)
        print(f"Tokenizer saved to {tokenizer_save_path}")