from transformers import CLIPProcessor, CLIPModel
from gensim import corpora
from gensim.summarization import bm25
import os, random, json, pickle, base64
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter, defaultdict
import json
import requests
from PIL import Image
from io import BytesIO

# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../3-13 classifier'))
# from q_classify_difficulty_type_analysis import evaluate_model

dataset = json.load(open("../../WebQA_train_val.json", "r"))
test_dataset = json.load(open("../../predicted_test_with_qtype.json", "r"))
pred_label_path = '/home/keerx/WebQA_Baseline/vlp/3-13 classifier/pred_labels.txt'

with open(pred_label_path, "r") as f:
    pred_labels = [line.strip() for line in f]

val_dataset = {k: dataset[k] for k in dataset if dataset[k]['split'] == 'val'}
test_dataset = {k: test_dataset[k] for k in test_dataset}

# print(Counter([dataset[k]['split'] for k in dataset]))
# print(len(set([dataset[k]['Guid'] for k in dataset])))
# print(Counter([dataset[k]['Qcate'] for k in dataset]))

# Load VLP model (using CLIP as an example of a multimodal VLP model)
device = "cuda" if torch.cuda.is_available() else "cpu"
vlp_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model, preprocess = clip.load("ViT-B/16", device=device)

''' with open("../../imgs.lineidx", "r") as fp_img:
#     img_lineidx = [int(i.strip()) for i in fp_img.readlines()]
#     print(len(img_lineidx))

# def get_image_input(i):
#     with open('../../imgs.tsv', "r") as fp_img:
#         fp_img.seek(img_lineidx[i])
#         imgid, img_base64 = fp_img.readline().strip().split('\t')
#     image = Image.open(BytesIO(base64.b64decode(img_base64)))
#     image_input = preprocess(image)
#     return image_input'''

def get_dynamic_retrieval_count(predicted_num, total_candidates, gold_context_num=None, 
                               min_threshold=2, max_threshold=None, scaling_factor=2, 
                               min_percentage=0.1, max_percentage=0.3):
    """
    Dynamically determine the number of contexts to retrieve based on the prediction, 
    total candidates, and optional gold context information.
    
    Parameters:
    -----------
    predicted_num : int
        The number of contexts predicted to be needed
    total_candidates : int
        The total number of candidate contexts available
    gold_context_num : int, optional
        The actual number of gold contexts (if known during validation)
    min_threshold : int, default=2
        Minimum number of contexts to retrieve
    max_threshold : int, optional
        Maximum number of contexts to retrieve (defaults to half of total_candidates)
    scaling_factor : float, default=1.5
        Factor to scale the predicted number for safety
    min_percentage : float, default=0.1
        Minimum percentage of total candidates to retrieve
    max_percentage : float, default=0.5
        Maximum percentage of total candidates to retrieve
    
    Returns:
    --------
    int : The number of contexts to retrieve
    """
    # Set default max_threshold if not provided
    if max_threshold is None:
        max_threshold = int(total_candidates * max_percentage)
    
    # Calculate base retrieval count from prediction
    base_count = int(predicted_num * scaling_factor)
    
    # Ensure we're within percentage bounds of total candidates
    min_by_percentage = max(min_threshold, int(total_candidates * min_percentage))
    max_by_percentage = min(max_threshold, int(total_candidates * max_percentage))
    
    # If we know gold context count (validation scenario), adjust accordingly
    if gold_context_num is not None:
        # Add a safety margin to gold count
        adjusted_count = int(gold_context_num * 1.2)
        # Blend the predicted and adjusted counts
        base_count = max(base_count, adjusted_count)
    
    # Apply bounds
    retrieval_count = max(min_by_percentage, min(base_count, max_by_percentage))
    
    # Ensure we don't exceed total candidates
    retrieval_count = min(retrieval_count, total_candidates)
    
    return retrieval_count

def compute_retrieval_metrics(pred, gth):
    # print(f'prediction: {pred}, gth: {gth}')
    common = len(set(pred).intersection(gth))
    RE = common / (len(gth)) 
    PR = common / (len(pred)) # No protection against division by zero because it's assumed that CLIP never gives empty output
    F1 = 2*PR*RE / (PR + RE + 1e-5)
    return F1, RE, PR

# def compute_vlp_scores(query_text, candidate_texts, candidate_image_urls=None):
    """Compute VLP similarity scores between query and candidate texts."""
    text_inputs = processor(text=[query_text] + candidate_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    
    with torch.no_grad():
        text_features = vlp_model.get_text_features(**text_inputs)

    # Extract embeddings
    query_text_embedding = text_features[0]  
    candidate_text_embeddings = text_features[1:] 

    text_sim_scores = torch.nn.functional.cosine_similarity(query_text_embedding, candidate_text_embeddings)

    # If candidate images are provided, process them
    if candidate_image_urls:
        
        candidate_images = [load_image(url) for url in candidate_image_urls]

        # Process image inputs
        image_inputs = processor(images=candidate_images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        # Compute image embeddings
        with torch.no_grad():
            candidate_image_embeddings = vlp_model.get_image_features(**image_inputs)

        # Compute image similarity scores (query text ↔ candidate images)
        image_sim_scores = torch.nn.functional.cosine_similarity(query_text_embedding, candidate_image_embeddings)

        # Combine text and image similarity scores (if images exist)
        combined_scores = (text_sim_scores + image_sim_scores) / 2  # Simple average fusion
    else:
        # Only text scores if no images are provided
        combined_scores = text_sim_scores  

    return combined_scores.cpu().numpy()

def top_n_test():
    id_retrieved_contexts_strict = defaultdict(dict)
    id_retrieved_contexts_2times = defaultdict(dict)
    id_retrieved_contexts_dynamic = defaultdict(dict)


    for k, v in tqdm(test_dataset.items()):
        pred = v['predicted_type']
        pred_type = pred[:-1]
        pred_num = int(pred[-1])
        key = 'Qtxt' if pred_type == 'text' else 'Qimg'
        corpus = []
        corpus_text = []
        candidate_image_urls = []
        contexts = []

        if key == 'Qtxt':
            contexts.extend([f for f in v['txt_Facts']])
            corpus.extend([x['fact'].split() for x in v['txt_Facts']])
            corpus_text.extend([x['fact'] for x in v['txt_Facts']])
           
        else:
            contexts.extend([f for f in v['img_Facts']])
            corpus.extend([x['caption'].split() for x in v['img_Facts']])
            corpus_text.extend([x['caption'] for x in v['img_Facts']])
            candidate_image_urls.extend(x['imgUrl'] for x in v['img_Facts'])

        # import ipdb; 
        # ipdb.set_trace()
        dictionary = corpora.Dictionary(corpus)
        corpus = [dictionary.doc2bow(text) for text in corpus]
        bm25_obj = bm25.BM25(corpus)

        query_doc = dictionary.doc2bow(v['Q'].replace('"', '').split())
        scores = bm25_obj.get_scores(query_doc)
        dynamic_pred = get_dynamic_retrieval_count(predicted_num=pred_num, total_candidates=len(corpus))
        # print(f'predicted number is {pred_num}, dynamic pred is {dynamic_pred}')
        best_docs_strict = sorted(range(len(scores)), key=lambda i: scores[i])[-pred_num:]  # Top-N from BM25
        best_docs_2times = sorted(range(len(scores)), key=lambda i: scores[i])[-2*pred_num:]  # Top-N from BM25
        best_docs_dynamic = sorted(range(len(scores)), key=lambda i: scores[i])[-dynamic_pred:]  # Top-N from BM25

        best_contexts_strict = [contexts[i] for i in best_docs_strict]
        best_contexts_2times = [contexts[i] for i in best_docs_2times]
        best_contexts_dynamic = [contexts[i] for i in best_docs_dynamic]


        query_text = v['Q']  # The original question

        id_retrieved_contexts_strict[k] = {
            'Question': query_text,
            'Answer': v['A'],
            'Retrieved Contexts': best_contexts_strict
        }
        id_retrieved_contexts_2times[k] = {
            'Question': query_text,
            'Answer': v['A'],
            'Retrieved Contexts': best_contexts_2times
        }
        id_retrieved_contexts_dynamic[k] = {
            'Question': query_text,
            'Answer': v['A'],
            'Retrieved Contexts': best_contexts_dynamic
        }
    
    return id_retrieved_contexts_strict, id_retrieved_contexts_2times, id_retrieved_contexts_dynamic


def top_n():
    retricted_bm25_scores = {'Qimg': [], 'Qtxt': []}
    id_retrieved_contexts = defaultdict(dict)

    # retricted_vlp_scores = {'Qimg': [], 'Qtxt': []}
    accurate_qtype = 0
    accurate_qlength = 0
    for (k, v), label in tqdm(zip(val_dataset.items(), pred_labels)):
        pred_type = label[:-1]
        pred_num = int(label[-1])

        true_key = 'Qtxt' if v['Qcate'] == 'text' else 'Qimg'
        key = 'Qtxt' if pred_type == 'text' else 'Qimg'
        if true_key == key:
            accurate_qtype += 1
        corpus = []
        corpus_text = []
        candidate_image_urls = []
        contexts = []

        if key == 'Qtxt':
            contexts.extend([f for f in v['txt_posFacts']])
            corpus.extend([x['fact'].split() for x in v['txt_posFacts']])
            corpus_text.extend([x['fact'] for x in v['txt_posFacts']])
            if len(v['txt_posFacts']) == pred_num:
                accurate_qlength += 1
            ans = list(range(len(corpus)))

        else:
            contexts.extend([f for f in v['img_posFacts']])
            corpus.extend([x['caption'].split() for x in v['img_posFacts']])
            corpus_text.extend([x['caption'] for x in v['img_posFacts']])
            candidate_image_urls.extend(x['imgUrl'] for x in v['img_posFacts'])
            if len(v['img_posFacts']) == pred_num:
                accurate_qlength += 1
            ans = list(range(len(corpus)))

        if key == 'Qtxt':
            contexts.extend([f for f in v['txt_negFacts']])
            corpus.extend([x['fact'].split() for x in v['txt_negFacts']])
            corpus_text.extend([x['fact'] for x in v['txt_negFacts']])
        else:
            contexts.extend([f for f in v['img_negFacts']])
            corpus.extend([x['caption'].split() for x in v['img_negFacts']])
            corpus_text.extend([x['caption'] for x in v['img_negFacts']])
            candidate_image_urls.extend(x['imgUrl'] for x in v['img_negFacts'])

        # corpus.extend([x['fact'].split() for x in dataset[g]['txt_negFacts']])
        # corpus.extend([x['caption'].split() for x in dataset[g]['img_negFacts']])
        # corpus_text.extend([x['fact'] for x in dataset[g]['txt_negFacts']])
        # corpus_text.extend([x['caption'] for x in dataset[g]['img_negFacts']])

        # import ipdb; 
        # ipdb.set_trace()
        dictionary = corpora.Dictionary(corpus)
        corpus = [dictionary.doc2bow(text) for text in corpus]
        bm25_obj = bm25.BM25(corpus)

        query_doc = dictionary.doc2bow(v['Q'].replace('"', '').split())
        scores = bm25_obj.get_scores(query_doc)
        dynamic_pred = get_dynamic_retrieval_count(predicted_num=pred_num, total_candidates=len(corpus), gold_context_num=len(ans))
        # print(f'predicted number is {pred_num}, dynamic pred is {dynamic_pred}')
        best_docs = sorted(range(len(scores)), key=lambda i: scores[i])[-pred_num:]  # Top-N from BM25
        # best_docs = sorted(range(len(scores)), key=lambda i: scores[i])[-2*pred_num:]  # Top-N from BM25
        
        best_contexts = [contexts[i] for i in best_docs]

        # Step 2: Rerank using VLP (CLIP-based)
        query_text = v['Q']  # The original question

        id_retrieved_contexts[k] = {
            'Question': query_text,
            'Answer': v['A'],
            'Retrieved Contexts': best_contexts
        }

        # candidate_texts = [corpus_text[i] for i in best_docs]
        # vlp_scores = compute_vlp_scores(query_text, corpus_text, candidate_image_urls)
        # print(vlp_scores)
        # reranked_docs = [corpus_text[i] for i in vlp_scores.argsort()[::-1]][:2]  # Sort by VLP scores and select top2
        # reranked_docs = vlp_scores.argsort()[::-1][:2]
        retricted_bm25_scores[key].append(compute_retrieval_metrics(set(best_docs), set(ans)))
        # retricted_vlp_scores[key].append(compute_retrieval_metrics(set(reranked_docs), set(ans)))
    
    q_type_acc = accurate_qtype / len(val_dataset)
    q_len_acc = accurate_qlength / len(val_dataset)
    print("q classify type length accuracy: type acc = {:.4f}, length acc = {:.4f}".format(q_type_acc, q_len_acc))
    return retricted_bm25_scores, id_retrieved_contexts
    # return retricted_bm25_scores, retricted_vlp_scores


retricted_bm25_scores, id_retrieved_contexts = top_n()
# test_retrieved_contexts_strict, test_retrieved_contexts_2times, test_retrieved_contexts_dynamic = top_n_test()
# with open('test_retrieved_contexts_strict.json', 'w', encoding='utf-8') as f:
#     json.dump(test_retrieved_contexts_strict, f, ensure_ascii=False, indent=2)
# with open('test_retrieved_contexts_2times.json', 'w', encoding='utf-8') as f:
#     json.dump(test_retrieved_contexts_2times, f, ensure_ascii=False, indent=2)
# with open('test_retrieved_contexts_dynamic.json', 'w', encoding='utf-8') as f:
#     json.dump(test_retrieved_contexts_dynamic, f, ensure_ascii=False, indent=2)

print(len(retricted_bm25_scores['Qimg']), len(retricted_bm25_scores['Qtxt']))
# print(len(retricted_vlp_scores['Qimg']), len(retricted_vlp_scores['Qtxt']))
with open('val_retrieved_contexts.json', 'w', encoding='utf-8') as f:
    json.dump(id_retrieved_contexts, f, ensure_ascii=False, indent=2)

print("BM25 strict")
print("BM25 unknownM img queries: F1={:.4f}, RE={:.4f}, PR={:.4f}".format(np.mean([P[0] for P in retricted_bm25_scores['Qimg']]), np.mean([P[1] for P in retricted_bm25_scores['Qimg']]), np.mean([P[2] for P in retricted_bm25_scores['Qimg']]) ))
print("BM25 unknownM txt queries: F1={:.4f}, RE={:.4f}, PR={:.4f}".format(np.mean([P[0] for P in retricted_bm25_scores['Qtxt']]), np.mean([P[1] for P in retricted_bm25_scores['Qtxt']]), np.mean([P[2] for P in retricted_bm25_scores['Qtxt']]) ))
    
# print("BM25 unknownM img queries: F1={:.4f}, RE={:.4f}, PR={:.4f}".format(np.mean([P[0] for P in retricted_vlp_scores['Qimg']]), np.mean([P[1] for P in retricted_vlp_scores['Qimg']]), np.mean([P[2] for P in retricted_vlp_scores['Qimg']]) ))
# print("BM25 unknownM txt queries: F1={:.4f}, RE={:.4f}, PR={:.4f}".format(np.mean([P[0] for P in retricted_vlp_scores['Qtxt']]), np.mean([P[1] for P in retricted_vlp_scores['Qtxt']]), np.mean([P[2] for P in retricted_vlp_scores['Qtxt']]) ))
    
