import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from q_classify_difficulty_with_type import DetailedTypeDatasetWithContext, collate_fn, prepare_data_train_val


# Load Model and Tokenizer
MODEL_PATH = "saved_model_detailed_classification_longformer/longformer_difficulty_classification_5.pth"
TOKENIZER_PATH = "saved_model_detailed_classification_longformer/tokenizer_difficulty_5"
DATA_PATH = "../WebQA_train_val.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = LongformerTokenizer.from_pretrained(TOKENIZER_PATH)
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Load validation data
data = prepare_data_train_val(DATA_PATH, debugging_partition=False)  # set debugging_partition to True to sample 1000 instances
val_dataset = DetailedTypeDatasetWithContext(data, tokenizer, split='val')
val_loader = DataLoader(val_dataset, batch_size=3, collate_fn=collate_fn)

# Identify unique labels in validation set
val_labels = set()
for _, _, label in val_dataset:
    val_labels.add(label.item())
val_labels = sorted(list(val_labels))
label_names = ["text1", "text2", "text3", "text4", "text5", "img1", "img2", "img3"]
valid_label_names = [label_names[i] for i in val_labels]
print('Data loaded')

# Evaluate Model
all_preds, all_labels = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in tqdm(val_loader, desc="Evaluating"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
print('Model loaded')

# Calculate accuracy and confusion matrix
overall_accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds, labels=val_labels)
class_report = classification_report(all_labels, all_preds, target_names=valid_label_names)

# Print evaluation results
print(f"Overall Accuracy: {overall_accuracy:.4f}\n")
print("Classification Report:")
print(class_report)
print(confusion_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=valid_label_names, yticklabels=valid_label_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()