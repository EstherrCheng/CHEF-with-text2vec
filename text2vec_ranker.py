import torch
import os
import re
import sys
import copy
import random
import time, datetime
from time import sleep
import json, csv
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import numpy as np

import sys

sys.path.append('..')
from text2vec import SentenceModel, cos_sim, semantic_search


def main():
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # model = AutoModel.from_pretrained("bert-base-chinese")
    # model = model.to(device)

    data_list = json.load(open('/home/chengyuyang/text2vec/CHEF/CHEF_data/test.json', 'r', encoding='utf-8')) \
                + json.load(open('/home/chengyuyang/text2vec/CHEF/CHEF_data/train.json', 'r', encoding='utf-8'))
    similar_evs = []
    for row in tqdm(data_list):
        claim = row['claim']
        ev_sents = []
        # ev_sents += re.split(r'[？：。！（）.“”…\t\n]', row['content'])
        for ev in row['evidence'].values():
            ev_sents += re.split(r'[？。！“”…\t\n]', ev['text'])
        ev_sents = [sent for sent in ev_sents if len(sent) > 5]
        sent2sim = {}
        hits = textSimilarity(claim, ev_sents)
        for hit in hits[0]:
            sent2sim[ev_sents[hit['corpus_id']]] = hit['score']
        sent2sim = list(sent2sim.items())
        sent2sim.sort(key=lambda s: s[1], reverse=True)
        similar_evs.append([[s[0], s[1]] for s in sent2sim[:5]])
        with open('text2vec_result.jsonl', 'a+', encoding='utf-8') as f:
            tmp = json.dumps([[s[0], s[1]] for s in sent2sim[:5]], ensure_ascii=False)
            print(tmp, file=f)


def textSimilarity(claim, ev_sents):
    embedder = SentenceModel()
    claim_embeddings = embedder.encode(claim)
    # corpus_embeddings = embedder.encode(corpus)
    evsents_embeddings = embedder.encode(ev_sents)
    hits = semantic_search(claim_embeddings, evsents_embeddings, top_k=5)
    return hits
    # print("\n\n======================\n\n")
    # print("Query:", query)
    # print("\nTop 5 most similar sentences in corpus:")
    # hits = hits[0]  # Get the hits for the first query
    # for hit in hits:
    # print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


def tryMaxLen(tokenizer, model, sentences):
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print(f"max sentence length = {max_len}")
    return max_len


if __name__ == '__main__':
    main()

