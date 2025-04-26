# OMQA-11797
This project aims to optimize multimodal QA by analyzing the question first. By classifying a question based on the number of context required to answer it and the modality of the contexts, our proposed pipeline can do better in the [WebQA](https://github.com/WebQnA/WebQA) dataset.

# Question Analyzer
The question analyzer is trained using `q_classify_difficulty_with_type.py`. It is specifically for the WebQA dataset, so its output is limited to the following question types: text1, text2, text3, text4, text5, img1, img2, img3, where text1 means a text-only question requiring 1 context to answer, etc.

To analyze a trained question classifier, you may find `q_classify_difficulty_type_analysis.py` useful.

The predicted question types for the test set of WebQA can be found [here](https://drive.google.com/file/d/1MBpHxSHbrtteHNyurAnk7sJGuMgn8LvE/view?usp=sharing).
# Context Retriever
The context retriever part is achieved using code in both `BM25_full_retrieval.ipynb` and `BM25+VLP.py`. The main code is adapted  from the original WebQA repository, and the original code based can be found [here](https://github.com/WebQnA/WebQA_Baseline/blob/main/vlp/BM25_retrieval/BM25_full_retrieval.ipynb).

# Answer Generator
