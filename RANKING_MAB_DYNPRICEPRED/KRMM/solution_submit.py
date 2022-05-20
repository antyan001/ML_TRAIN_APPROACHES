import os

import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F

import faiss
from langdetect import detect, DetectorFactory
import json
import pickle
import requests
from flask import Flask, request, session, jsonify
from time import sleep

EMB_PATH_GLOVE = os.environ.get("EMB_PATH_GLOVE")
EMB_PATH_KNRM  = os.environ.get("EMB_PATH_KNRM")
VOCAB_PATH     = os.environ.get("VOCAB_PATH")
MLP_PATH       = os.environ.get("MLP_PATH")


app = Flask(__name__)
app.secret_key = 'mykey'
app.config['SESSION_TYPE'] = 'filesystem'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1., requires_grad = False):
        super().__init__()
        mu_ = np.array(mu)
        sigma_ = np.array(sigma)
        self.requires_grad = requires_grad
        self.mu = torch.nn.Parameter(torch.Tensor(mu_), requires_grad=self.requires_grad)
        self.sigma = torch.nn.Parameter(torch.Tensor(sigma_), requires_grad=self.requires_grad)

    def forward(self, x):
        adj = x - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)        

        
class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool = False, kernel_num: int = 30,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [15, 7]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:

        mus = [1.0]
        if self.kernel_num > 1:
            bin_size = 2.0 / (self.kernel_num - 1)  
            mus.append(1 - bin_size / 2)
            for i in range(1, self.kernel_num - 1):
                mus.append(mus[i] - bin_size)
        mus = list(reversed(mus))
        sigmas = [self.sigma] * (self.kernel_num - 1) + [self.exact_sigma]  
        
        gausskern_lst = [(GaussianKernel(mu,sigma)) for mu, sigma in zip(mus, sigmas)]
        kernels = torch.nn.ModuleList(gausskern_lst)
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:        
        if self.out_layers:
            output = []
            hidden_sizes = [self.kernel_num] + self.out_layers + [1]
            for i, hidden in enumerate(hidden_sizes[1:],1):
                output.append(torch.nn.ReLU())
                output.append(torch.nn.Linear(hidden_sizes[i-1], hidden))
        else:
            output = [torch.nn.Linear(self.kernel_num, 1)]
        return torch.nn.Sequential(*output)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        query = self.embeddings(query)
        doc = self.embeddings(doc)
        query = query / (query.norm(p=2, dim=-1, keepdim=True) + 1e-16)
        doc = doc / (doc.norm(p=2, dim=-1, keepdim=True) + 1e-16)
        return torch.bmm(query, doc.transpose(-1, -2))

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)
                
        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left, Embedding], [Batch, Right, Embedding]
        query, doc = inputs['query'], inputs['document']
        
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out
    
    
def collate_fn(batch_objs: List[List[int]], name: str = 'query'):
    max_len_q1 = -1

    is_triplets = False
    for elems in batch_objs:
        max_len_q1 = max(len(elems), max_len_q1)

    q1s = []

    for elems in batch_objs:
        pad_len1 = max_len_q1 - len(elems)

        q1s.append(elems + [0] * pad_len1)

    ret_left = {name: torch.LongTensor(q1s)}

    return ret_left

def hadle_punctuation(inp_str: str) -> str:
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    out = regex.sub('', inp_str)
    return out

def simple_preproc(inp_str: str) -> List[str]:
    rem_puct_str = hadle_punctuation(inp_str).lower()       
    return nltk.word_tokenize(rem_puct_str)

def tokenized_text_to_index(tokenized_text: List[str]) -> List[int]:
    return [vocab.get(item, vocab['OOV']) for item in tokenized_text]

def avg_emb(toks):
    np.random.seed(17)
    dim_of_embed = len(glove_emb.get('the'))
    zeros = np.zeros_like(glove_emb.get('the'))
    unk_embedding = np.random.uniform(-0.5, 0.5, dim_of_embed)  #np.random.uniform(-1.5, 1.5, dim_of_embed) #emb_matrix["weight"][1].numpy() 
    avg_emb = zeros #unk_embedding
    
    if len(toks) != 0:
        if toks[0] is None: return None
        for tok in toks:
            avg_emb += glove_emb.get(tok, unk_embedding)
        return avg_emb / len(toks) 
    else:
        return unk_embedding
   
def _read_glove_embeddings(file_path: str) -> Dict[str, List[str]]:
    res = dict([tuple(line.split(" ", 1)) for line in open(file_path, 'r')]) 
    fin = dict([( k, list(map(float,v.split(" "))) ) for k,v in res.items() if k not in string.punctuation])
    return fin  

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

with open(VOCAB_PATH, 'r') as fin:
    vocab = json.load(fin)

emb_matrix = torch.load(EMB_PATH_KNRM)
knrm = KNRM(emb_matrix["weight"], freeze_embeddings=True)
mlp = torch.load(MLP_PATH)
knrm.mlp = mlp

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

@app.route('/ping')
def ping():
    if knrm and vocab:
    	return jsonify({"status": "ok"})
    else: 
        return jsonify({"status": None})


@app.route('/update_index', methods=['POST'])
def update_index():
    
    global index, documents, doc_tokens, glove_emb    
    glove_emb = _read_glove_embeddings(EMB_PATH_GLOVE)    
       
    try:  
        documents  = json.loads(request.json) if issubclass(request.json.__class__, str) else request.json
        doc_tokens = np.array([(q,simple_preproc(text)) for q,text in documents["documents"].items()], dtype=object)
        doc_emb    = [(q,avg_emb(toks)) for q,toks in doc_tokens]
        vectors    = np.array([emb for _,emb in doc_emb], dtype=object).astype('float32')
        
        #norm = np.linalg.norm(vectors, axis = 1, keepdims = True)
        #norm_vec = vectors / norm

        k = 34500 #34500 #int(len(vectors)/15)

        dim = len(glove_emb.get('the'))
        code_size = 5
        quantiser = faiss.IndexFlatL2(dim) 
        #quantizer = faiss.IndexFlatL2(dim)
        #index = faiss.IndexIVFPQ(quantizer, dim, k, code_size, 8)    	
        index = faiss.IndexIVFFlat(quantiser, dim, k)
        #index = faiss.index_factory(dim, 'IVF%s, Flat' % k,  faiss.METRIC_L2) #METRIC_INNER_PRODUCT
        index.train(vectors)     
        index.add(vectors)   
    
        return jsonify({"status": "ok", "index_size": len(doc_tokens)}) #index.ntotal
    except:
        return jsonify({"status": None, "index_size": None})


@app.route('/query', methods=['POST'])
def query():
    
    lang_check = []
    fin_rel = []
    DetectorFactory.seed = 0
    
    query  = json.loads(request.json) if issubclass(request.json.__class__, str) else request.json
        
    if 'index' in globals():  
        lang_check = [True if detect(text)=='en' else False for text in query['queries']]
        query_tokens = np.array([simple_preproc(text) if lang_check[i] == True else [None] 
                                 for i, text in enumerate(query['queries'])], dtype=object)    
        query_emb    = np.array([avg_emb(toks) for toks in query_tokens], dtype=object)
#         query = query_emb.astype('float32')   
            
        for i,q in enumerate(query_emb):
            if q is not None:    
            
                index.nprobe = 12 #12
                topn = 10 #10 #index.ntotal
                
                #q_norm  = q / np.linalg.norm(q, axis = 0, keepdims = True)
                
                D, I = index.search(np.expand_dims(q,axis=0).astype("float32"), topn)      
                indx = I[(I>-1)]
                toks_to_idx = [tokenized_text_to_index(tok) for _,tok in doc_tokens[indx]]
                padded_toks_to_idx = collate_fn(toks_to_idx, name = 'document')

                toks_to_idx = [tokenized_text_to_index(tok) for tok in np.expand_dims(query_tokens[i],axis=0)]*len(indx)
                padded_toks_to_idx.update(collate_fn(toks_to_idx, name = 'query'))

                preds = knrm.predict(padded_toks_to_idx)
                ind_preds = preds.squeeze(1).sort(descending=True)[1].tolist()
                rel_doc_ids = [k for k, _ in doc_tokens[indx][ind_preds]]

                res = [tuple([k,text]) for k,text in documents["documents"].items() if k in rel_doc_ids][:10]
                fin_rel.append(res)
            else:
                fin_rel.append(None)
                
        return jsonify({"lang_check": lang_check, "suggestions": fin_rel})            
    else:
        return jsonify({"status": "FAISS is not initialized!"})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='11000', debug=True)
    
    
    
   