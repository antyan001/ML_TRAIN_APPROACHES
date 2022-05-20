import math
from math import log2
import torch
from torch import Tensor, sort

def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor, is2Darr: bool = True) -> int:

    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64)
        ys_pred = ys_pred.reshape((-1,1)).type(torch.float64)
    
    input_shape = ys_pred.shape[0]
    
    cat = torch.hstack((ys_pred, ys_true))
    t_sorted = cat[cat[:, 0].sort(descending=True)[1]]
    t_sorted = t_sorted.unsqueeze(0)
    
    input_tensor = t_sorted    

    first = input_tensor.repeat(1, input_shape, 1)
    second = input_tensor.unsqueeze(2)
    second = second.repeat(1,1,input_shape,1).view(input_tensor.size(0),-1,input_tensor.size(2))
    output_tensor = torch.cat((first,second), dim=2) 
    
    fin = output_tensor.view(output_tensor.size(1), output_tensor.size(2))
    res = fin[fin[:, 0].sort(descending=True)[1]]
    res_noneq = res[(~(res[:,0]==res[:,2]))&(~(res[:,1]==res[:,3]))]
    res_noneq = res_noneq.view(-1,2,2) 

    res_eq = res[(~(res[:,0]==res[:,2]))&((res[:,1]==res[:,3]))]
    res_eq = res_eq.view(-1,2,2) 
        
    tot = [1 if ((row[0, 1]<row[1, 1])&(row[0, 0]<row[1, 0]))|
                ((row[0, 1]>row[1, 1])&(row[0, 0]>row[1, 0]))
           else 0 for row in res_noneq
          ]
    
    cnt_conc = sum(tot)
    cnt_disc = len(tot) - sum(tot)
    
    return cnt_disc // 2


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2.**y_value - 1.
    else:
        return y_value
    
    
def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor,
        gain_scheme: str, is2Darr: bool = True) -> float:

    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64) 
        ys_pred = ys_pred.reshape((-1,1)).type(torch.float64)     
    
    input_shape = ys_pred.shape[0]
    cat = torch.hstack((ys_pred, ys_true))
    t_sorted = cat[cat[:, 0].sort(descending=True)[1]]
#     t_sorted = t_sorted.unsqueeze(0)

    input_tensor = t_sorted.numpy()

    factors = torch.Tensor([
                            compute_gain(y_value = float(input_tensor[i,1]), 
                                         gain_scheme = gain_scheme
                                        )/math.log2(i+2)
                            for i in range(input_tensor.shape[0])
                           ]
                          ).type(torch.float64)

    return float(torch.sum(factors,dim=0).numpy())     
    
    
def dcg_k(ys_true: torch.Tensor, ys_pred: torch.Tensor,
          ndcg_top_k: int, gain_scheme: str, is2Darr: bool = True) -> float:

    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64) 
        ys_pred = ys_pred.reshape((-1,1)).type(torch.float64) 
    
    input_shape = ys_pred.shape[0]
    cat = torch.hstack((ys_pred, ys_true))
    t_sorted = cat[cat[:, 0].sort(descending=True)[1]]
#     t_sorted = t_sorted.unsqueeze(0)

    input_tensor = t_sorted[:ndcg_top_k].numpy()

    factors = torch.Tensor([
                            compute_gain(y_value = float(input_tensor[i,1]), 
                                         gain_scheme = gain_scheme
                                        )/math.log2(i+2)
                            for i in range(input_tensor.shape[0])
                           ]
                          ).type(torch.float64)

    return float(torch.sum(factors,dim=0).numpy())    


def ndcg_k(ys_true: torch.Tensor, ys_pred: torch.Tensor,
            ndcg_top_k: int, gain_scheme: str) -> float:

    _dcg_k = dcg_k(ys_true, ys_pred, ndcg_top_k, gain_scheme)

    _idcg_k = dcg_k(ys_true, ys_true, ndcg_top_k, gain_scheme)

    res = _dcg_k/_idcg_k

    return res      


def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor,
         gain_scheme: str) -> float:

    _dcg = dcg(ys_true, ys_pred, gain_scheme)

    _idcg = dcg(ys_true, ys_true, gain_scheme)

    res = _dcg/_idcg

    return res 


def idcg_k(ys_true: torch.FloatTensor, 
           gain_scheme: str, ndcg_top_k: int, is2Darr: bool = True) -> float:

    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64)

    input_shape = ys_true.shape[0]

    t_sorted = ys_true.sort(descending=True, axis=0)[0][:ndcg_top_k]

    input_tensor = t_sorted.numpy()

    factors = torch.Tensor([
                            compute_gain(y_value = input_tensor[i], 
                                         gain_scheme = gain_scheme
                                        )/math.log2(i+2)
                            for i in range(input_tensor.shape[0])
                           ]
                          ).type(torch.float64)    

#         idcg = torch.cusum(factors,dim=0)[-1]
    idcg_k = torch.sum(factors,dim=0)

    res = float(idcg_k.numpy())

    return res


def idcg(ys_true: torch.FloatTensor, gain_scheme: str, is2Darr: bool = True) -> float:

    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64)

    input_shape = ys_true.shape[0]

    t_sorted = ys_true.sort(descending=True, axis=0)[0]

    input_tensor = t_sorted.numpy()

    factors = torch.Tensor([
                            compute_gain(y_value = input_tensor[i], 
                                              gain_scheme = gain_scheme
                                             )/math.log2(i+2)
                            for i in range(input_tensor.shape[0])
                           ]
                          ).type(torch.float64)    

#         idcg = torch.cusum(factors,dim=0)[-1]
    idcg = torch.sum(factors,dim=0)

    res = float(idcg.squeeze(0).numpy())

    return res

    
def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int, is2Darr: bool = True) -> float:

    def first_nonzero(x, axis=0):
        nonz = (x > 0)
        return ((nonz.cumsum(axis) == 1) & nonz).max(axis)    

    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64)
        ys_pred = ys_pred.reshape((-1,1)).type(torch.float64)
    
    input_shape = ys_pred.shape[0]
    sum1 = int(ys_true.sum().numpy())
    kk = min(k, sum1) #min(k, input_shape)
    
    cat = torch.hstack((ys_pred,ys_true))
    t_sorted = cat[cat[:, 0].sort(descending=True)[1]]
    
    ref = t_sorted[:,1].unsqueeze(1)
#     input_tensor = t_sorted
        
    # truncate repetitive zero elements in ys_true in order to 
    # maximize precission_at_k function
    # frst_nonzero_i = torch.argmax(t_sorted[:,1], 0, keepdim=False)
    
    frst_nonzero_i = int(first_nonzero(ref)[-1][0].numpy())
    input_tensor = t_sorted[frst_nonzero_i:]
        
    _sum1 = torch.sum(input_tensor[:kk,1]).numpy()
    
    if _sum1 != 0:
        res = _sum1/float(kk)
    else:
        res = -1
    return res


def average_precision(ys_true: Tensor, ys_pred: Tensor, is2Darr: bool = True) -> float:

    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64)
        ys_pred = ys_pred.reshape((-1,1)).type(torch.float64)
            
    input_shape = ys_pred.shape[0]
    
    cat = torch.hstack((ys_pred,ys_true))
    t_sorted = cat[cat[:, 0].sort(descending=True)[1]]
#     t_sorted = t_sorted.unsqueeze(0)
    
    input_tensor = t_sorted
 
    _total_sum1 = torch.sum(input_tensor[:,1]).numpy()
    if _total_sum1 == 0:
        return -1
    
    _sumatk = torch.Tensor([float(input_tensor[k,1] == 1.)* \
                            (torch.sum(input_tensor[:k+1,1])/(k+1)) for k in range(input_shape)])
    
    pr = torch.sum(_sumatk).numpy()

    res = pr/_total_sum1

    return res
    

def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor, is2Darr: bool = True) -> float:
    
    if not is2Darr:
        assert ys_true.dim() == 1
        assert ys_pred.dim() == 1    

        ys_true = ys_true.reshape((-1,1)).type(torch.float64)
        ys_pred = ys_pred.reshape((-1,1)).type(torch.float64)
    
    input_shape = ys_pred.shape[0]
    
    cat = torch.hstack((ys_pred,ys_true))
    t_sorted = cat[cat[:, 0].sort(descending=True)[1]]
#     t_sorted = t_sorted.unsqueeze(0)
    
    indices = torch.argmax(t_sorted[:,1], 0, keepdim=False)
    max_indx = indices.numpy()+1
    try:
        return 1./max_indx
    except:
        return None    
    
    
def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
 
    def plook(ind, rels, p_break):
        if ind == 0:
            return 1
        return plook(ind-1, rels, p_break)*(1-rels[ind-1])*(1-p_break)
    
    assert ys_true.dim() == 1
    assert ys_pred.dim() == 1
    
    input_shape = ys_pred.shape[0]
    ys_pred = ys_pred.reshape((-1,1)).type(torch.float32)
    ys_true = ys_true.reshape((-1,1)).type(torch.float32)
    
    cat = torch.hstack((ys_pred,ys_true))
    t_sorted = cat[cat[:, 0].sort(descending=True)[1]]    
    
    p_found = 0.
    for i in range(input_shape):
        prel = float(t_sorted[i,1].numpy())
        p_found+=plook(i, t_sorted[:,1], p_break)*prel
        
    return float(p_found.numpy())  





    
    
    
    
    
    