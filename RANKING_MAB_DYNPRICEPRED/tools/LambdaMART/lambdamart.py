#!/usr/local/bin/python3.7

import os
import shutil
import math
import pickle
import random
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import torch
import sklearn
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

from multiprocessing import Process, JoinableQueue
from queue import Queue
from threading import Thread
from joblib import Parallel, delayed
from multiprocessing import Pool
from pathlib import Path
from tqdm.notebook import tqdm
import itertools
import json
import time
import csv


def saver(q):
    file_path      = Path.joinpath(Path(os.getcwd()), 'csv', 'bestparams.csv')
    headers = ['lr', 'ndcg', 'dtree_params']
    #if not os.path.isfile(str(file_path)):
    with open(file_path, 'a', encoding='utf8') as outcsv:

        writer = csv.writer(outcsv, delimiter=',', quotechar='"', 
                            quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

        file_is_empty = os.stat(str(file_path)).st_size == 0
        if file_is_empty:
            writer.writerow(headers)     
        while True:
            strfrom_q = q.get()
            if strfrom_q is None: break
            lr, ndcg, json_params = strfrom_q.split('#')
            item = [lr, ndcg, json_params]                
            writer.writerow(item)                                    
            q.task_done()
        # Finish up
        q.task_done()   

    
def parse_pool(q, query_item, total_comb):
    
    pbar = tqdm(total=total_comb)

    while True: 
        for params in query_item:
            for _lr in np.array([0.05, 0.1, 0.5, 0.8]):
                sol = Solution(lr = _lr, **params)
                sol.fit()

                restr = json.dumps(sol.trees[0].get_params())
                ndcg = "{:10.6f}".format(sol.best_ndcg)
                q.put(str(_lr) + '#' + ndcg + '#' + restr)

            pbar.update(1)

            del sol
            
        break
    pbar.close()  
    

class Solution:
    def __init__(self, 
                 n_estimators: int = 50, 
                 lr: float = 0.6, 
                 ndcg_top_k: int = 10,
                 subsample: float = 1.0, 
                 colsample_bytree: float = 1.0,
                 loadfromlocal: bool = True,
                 train_name = "msrank_train.csv",
                 test_name  = "msrank_test.csv",
                 **kwarg):
        
        self.eps = 1.e-11       
        self.std_sc = StandardScaler()
        self.max_sc = MaxAbsScaler() 
        self.trees = []
        self.ndcg_lst = []
        self.tree_cols_indx = []
        self.best_ndcg = None
        
        self.gain_scheme = 'exp2'
        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.loadfromlocal = loadfromlocal
        self.train_dataset_name = train_name
        self.test_dataset_name = test_name

        self._prepare_data()
        self.num_samples,  self.num_input_features = self.X_train.shape
        
#         self.max_depth = max_depth
#         self.min_samples_leaf = min_samples_leaf

        self.tree_params = kwarg  
        
    def _groups_count_vectorizer(self, inp_query_ids: np.ndarray) -> Dict:
        uniq_indx={}
        for ind in inp_query_ids:
            if ind not in uniq_indx:
                uniq_indx[ind]=1
            else:
                uniq_indx[ind]+=1
                
        return uniq_indx
        
    def _get_data(self) -> List[np.ndarray]:
                
        if not self.loadfromlocal:
            train_df, test_df = msrank_10k()
        else:
            train_df = pd.read_csv(os.path.join("./csv", self.train_dataset_name), header=None)
            test_df  = pd.read_csv(os.path.join("./csv", self.test_dataset_name), header=None)

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, 
                                                            self.query_ids_train))
        self.X_test  = torch.FloatTensor(self._scale_features_in_query_groups(X_test, 
                                                            self.query_ids_test))
        
        self.ys_train = torch.FloatTensor(y_train).view(-1,1)
        self.ys_test  = torch.FloatTensor(y_test).view(-1,1)
        
        
    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
      
        uniq_indx =self._groups_count_vectorizer(inp_query_ids)

        shift = 0
        for ind, cnt in uniq_indx.items():

            sample = inp_feat_array[shift: shift+cnt]
            std_lst = sample.std(axis=0)
            const_cols = np.where(abs(std_lst) <= self.eps)[0]

            mask = np.zeros(sample.shape[-1], dtype=bool)
            mask[const_cols] = True   
            sample[:, mask] = self.max_sc.fit_transform(sample[:,mask])

            if len(const_cols) < sample.shape[-1]:
                mask = np.ones(sample.shape[-1], dtype=bool)
                mask[const_cols] = False  
                sample[:, mask] = self.std_sc.fit_transform(sample[:,mask])

            inp_feat_array[shift: shift+cnt] = sample

            shift+=cnt
                                                            
        return inp_feat_array 


    def compute_gain(self, y_value: float, gain_scheme: str) -> float:
        if gain_scheme == 'const':
            return y_value
        elif gain_scheme == 'exp2':
            return 2.**y_value - 1.
        else:
            return y_value
    
    def dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int, gain_scheme: str) -> float:
 
        input_shape = ys_pred.shape[0]
        cat = torch.hstack((ys_pred, ys_true))
        t_sorted = cat[cat[:, 0].sort(descending=True)[1]]
    #     t_sorted = t_sorted.unsqueeze(0)

        input_tensor = t_sorted[:ndcg_top_k].numpy()

        factors = torch.Tensor([
                                self.compute_gain(y_value = float(input_tensor[i,1]), 
                                             gain_scheme = gain_scheme)/math.log2(i+2)
                                for i in range(input_tensor.shape[0])
                               ]
                              ).type(torch.float64)

        return float(torch.sum(factors,dim=0).numpy())    

    
    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
            
        _dcg_k = self.dcg_k(ys_true, ys_pred, ndcg_top_k, self.gain_scheme)
                       
        _idcg_k = self.idcg_k(ys_true, self.gain_scheme, ndcg_top_k)

        res = _dcg_k/_idcg_k
        
        return res      

    def idcg_k(self, ys_true: torch.FloatTensor, 
               gain_scheme: str, ndcg_top_k: int, is2Darr: bool = True) -> float:

        if not is2Darr:
            assert ys_true.dim() == 1
            assert ys_pred.dim() == 1    

            ys_true = ys_true.reshape((-1,1)).type(torch.float64)

        input_shape = ys_true.shape[0]

        t_sorted = ys_true.sort(descending=True, axis=0)[0][:ndcg_top_k]

        input_tensor = t_sorted.numpy()

        factors = torch.Tensor([
                                self.compute_gain(y_value = input_tensor[i], 
                                                  gain_scheme = gain_scheme
                                                 )/math.log2(i+2)
                                for i in range(input_tensor.shape[0])
                               ]
                              ).type(torch.float64)    

#         idcg = torch.cusum(factors,dim=0)[-1]
        idcg_k = torch.sum(factors,dim=0)
        
        res = float(idcg_k.numpy())
        
        return res
    
    
    def idcg(self, ys_true: torch.FloatTensor, gain_scheme: str, is2Darr: bool = True) -> float:

        if not is2Darr:
            assert ys_true.dim() == 1
            assert ys_pred.dim() == 1    

            ys_true = ys_true.reshape((-1,1)).type(torch.float64)

        input_shape = ys_true.shape[0]

        t_sorted = ys_true.sort(descending=True, axis=0)[0]

        input_tensor = t_sorted.numpy()

        factors = torch.Tensor([
                                self.compute_gain(y_value = input_tensor[i], 
                                                  gain_scheme = gain_scheme
                                                 )/math.log2(i+2)
                                for i in range(input_tensor.shape[0])
                               ]
                              ).type(torch.float64)    

#         idcg = torch.cusum(factors,dim=0)[-1]
        idcg = torch.sum(factors,dim=0)
        
        res = float(idcg.squeeze(0).numpy())
        
        return res
    
    def compute_labels_in_batch(self, y_true):

        rel_diff = y_true - y_true.t()
        pos_pairs = (rel_diff > 0).type(torch.float32)
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        
        return Sij

    
    def compute_gain_diff(self, y_true, gain_scheme):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff    
    
    
    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        
        ideal_dcg = self.idcg(y_true, gain_scheme=self.gain_scheme)
        
        try:
            N = 1 / ideal_dcg
        except:
            N = 1.
            
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            Sij = self.compute_labels_in_batch(y_true)
            gain_diff = self.compute_gain_diff(y_true, self.gain_scheme)
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update


    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        
        uniq_indx = self._groups_count_vectorizer(queries_list)
        with torch.no_grad(): 
            
            shift = 0
            ndcgs = []
            for ind, cnt in uniq_indx.items():

                batch_preds = preds[shift: shift+cnt]
                batch_true = true_labels[shift: shift+cnt]
                ndcg_score = self._ndcg_k(batch_true, batch_preds, self.ndcg_top_k)            
                ndcgs.append(ndcg_score)

                shift+=cnt
            
            res = np.mean(ndcgs) 
        
        return res
 

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        
        if cur_tree_idx == 1:
            self.seed = cur_tree_idx
  
        uniq_indx = self._groups_count_vectorizer(self.query_ids_train)
    
        with torch.no_grad(): 
            
            shift = 0
            lambda_update = torch.Tensor().view(-1,1)
            for ind, cnt in uniq_indx.items(): 
                
                batch_x = self.X_train[shift: shift+cnt]
                batch_y = self.ys_train[shift: shift+cnt]
                tr_preds = train_preds[shift: shift+cnt]
                lambda_batch = self._compute_lambdas(batch_y, tr_preds)
                lambda_update = torch.vstack((lambda_update,lambda_batch)) 
                
                shift+=cnt
            
#             print(f"one tree --> lambdas min {lambda_update.min()} | max {lambda_update.max()}")
            
            rand_samples =  torch.randperm(self.num_samples)[:int(self.subsample*self.num_samples)]
    
            rand_colsample = torch.randperm(self.num_input_features)[:int(self.colsample_bytree*self.num_input_features)]
        
            tr_subset = self.X_train[rand_samples,:][:,rand_colsample].numpy().astype(np.float64)
            lambdas_subset = lambda_update[rand_samples].numpy().astype(np.float64)
            
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            
            X = imp_mean.fit_transform(tr_subset)
            y = lambdas_subset #imp_mean.fit_transform(lambdas_subset) 
                        
            dtree = DecisionTreeRegressor( 
                                            random_state = self.seed,
                                            **self.tree_params
                                         )

            dtree.fit(X, y)

            return dtree, rand_colsample.numpy() 

        
    def _update_terminal_regions(self, tree, X, y, lambdas, y_pred,
                                 sample_mask = None):
        
        terminal_regions = tree.apply(X)
        masked_terminal_regions = terminal_regions.copy()
    #     masked_terminal_regions[~sample_mask] = -1

        for leaf in np.where(tree.tree_.children_left ==
                             sklearn.tree._tree.TREE_LEAF)[0]:
            terminal_region = np.where(masked_terminal_regions == leaf)
            suml = np.sum(lambdas[terminal_region])
    #         sumd = np.sum(deltas[terminal_region])
    #         tree.value[leaf, 0, 0] = 0.0 if sumd == 0.0 else (suml / sumd)
            tree.tree_.value[leaf, 0, 0] = 0.0 if suml == 0.0 else suml

        y_pred += tree.tree_.value[terminal_regions, 0, 0] * self.lr
        
        return y_pred, tree


    def fit(self):
        
        np.random.seed(0)
        
        ##train cum preds
        lamb_preds_tr = torch.zeros_like(self.ys_train) + self.eps
        ## val cum preds
        results = torch.zeros_like(self.ys_test) + self.eps
        
        for i in range(1, self.n_estimators+1):
            
            dtree, cols_indx = self._train_one_tree( cur_tree_idx = i,
                                                     train_preds = lamb_preds_tr
                                                   )  
            
            X = self.X_train[:,cols_indx].numpy()
#             y = self.ys_train[cols_indx].numpy()
            lamb_update_tr = dtree.predict(X)
            lamb_preds_tr -= torch.Tensor(lamb_update_tr[:,np.newaxis])*self.lr
            
            self.trees.append(dtree)
            self.tree_cols_indx.append(cols_indx)
            
#             lamb_preds_tr, dtree = self._update_terminal_regions(tree = dtree, 
#                                                                  X = X, 
#                                                                  y = y, 
#                                                                  lambdas = lamb_update_tr, 
#                                                                  y_pred = lamb_preds_tr.view(-1).numpy())                   
#             lamb_preds_tr = torch.Tensor(lamb_preds_tr).view(-1,1)


            X_ts = self.X_test[:,cols_indx]
            lamb_update_ts = dtree.predict(X_ts)  
            results -= self.lr * torch.Tensor(lamb_update_ts[:,np.newaxis])
            mean_ndcgs    = self._calc_data_ndcg( queries_list=self.query_ids_test,
                                                  true_labels=self.ys_test, 
                                                  preds=results
                                                )            

            self.ndcg_lst.append(mean_ndcgs)
#             print("##estimator: {} ## TEST --> mean_ndcg: {}".format(i, mean_ndcgs))
        
        self.best_tree_indx = np.argmax(self.ndcg_lst)
                
        if self.best_tree_indx > 0:
            self.trees = self.trees[:self.best_tree_indx]
            self.tree_cols_indx = self.tree_cols_indx[:self.best_tree_indx]
        else:
            pass
                
        self.best_ndcg = self.ndcg_lst[self.best_tree_indx] 
         
    

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        
        results = torch.zeros_like(self.ys_test) + self.eps
        
        for i, tree in enumerate(self.trees):
            X_ts = data[:,self.tree_cols_indx[i]]
            lamb_update_ts = tree.predict(X_ts)  
            results -= self.lr * torch.Tensor(lamb_update_ts[:,np.newaxis])
       
        return results.type(torch.float64) 


    def save_model(self, fname: str):
        """
        Saves the model into a ".lmart" file with the name given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to save

        """
        directory = "./pkl"

        if not os.path.exists(directory):
            os.makedirs(directory)

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(filename) or os.path.islink(filename):
                    os.unlink(filename)
                elif os.path.isdir(filename):
                    shutil.rmtree(filename)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (filename, e))
        
        pickle.dump(self, open('%s.lmart' % (fname), "wb"), protocol=2)

        
    def load_model(self, fname: str):
        """
        Loads the model from the ".lmart" file given as a parameter.
        Parameters
        ----------
        fname : string
            Filename of the file you want to load
        """
        
        model = pickle.load(open(fname , "rb"))
        self.lr = model.lr
        self.tree_cols_indx = model.tree_cols_indx
        self.trees = model.trees   
        self.best_tree_indx = model.best_tree_indx
        self.best_ndcg = model.best_ndcg
  

    
    
if __name__ == '__main__':
    
    splitter = ['best', 'random']
    max_depth = [None, 5, 8, 15]
    min_samples_split = [2, 3, 7, 10]
    min_samples_leaf = [1, 2, 4]
    min_weight_fraction_leaf = [0.0, 1.e-7, 1.e-4]
    max_features = ['auto', 'sqrt', 'log2']
    max_leaf_nodes = [None, 10, 30, 40]
    min_impurity_decrease = [0.0, 1.e-7, 1.e-4]
    ccp_alpha = [0.0, 1.e-6, 1.e-4]

    dct_params = {
                    'splitter': splitter,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'min_weight_fraction_leaf': min_weight_fraction_leaf, 
                    'max_features': max_features,
                    'max_leaf_nodes': max_leaf_nodes,
                    'min_impurity_decrease': min_impurity_decrease, 
                    'ccp_alpha': ccp_alpha
                 }    
    
    
    tree_keys = [key for key in dct_params.keys()]

    total_lst_of_params = np.array([])

    arr = [v for _, v in dct_params.items()]
    all_comb = list( itertools.product( *arr ) )

    total_lst_of_params = [dict(zip(tree_keys, comb))  for comb in all_comb]    

    splitter = np.array_split(total_lst_of_params, 60)
    
    result_queue = JoinableQueue() #Queue()
    p = Thread(target=saver, args=(result_queue,))    
    threadlst=[]
    p.start()
    # argparser = args
    # We create list of thread}s and pass shared queue to all of them.
    threadlst=[Thread(target=parse_pool, 
                      args=(result_queue, params_lst, len(params_lst))) for i, params_lst in enumerate(splitter)]
    # Starting threads...
    print("Start: %s" % time.ctime())
    for th in threadlst:
        th.start()
    # Waiting for threads to finish execution... 
    for th in threadlst:
        th.join() 
    print("End:   %s" % time.ctime())

    result_queue.put(None) # Poison pill
    p.join()     

    
    
    
    
    
    
    
    
    
    
    
    
    
    