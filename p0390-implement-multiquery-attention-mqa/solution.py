import numpy as np

def multiquery_attention(X: np.ndarray, W_queries: list[np.ndarray], W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    seq_len, d_model = X.shape
    num_heads = len(W_queries)
    d_k = W_key.shape[1]
    d_v = W_value.shape[1]
    
    W_Q_stacked = np.stack(W_queries, axis=0)
    
    Q = np.einsum('sd,hdk->hsk', X, W_Q_stacked)
    
    K = X @ W_key
    V = X @ W_value
    
    scores = np.einsum('hsk,lk->hsl', Q, K) / np.sqrt(d_k)
    
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    heads_out = np.einsum('hsl,lv->hsv', weights, V)

    heads_out = heads_out.transpose(1, 0, 2).reshape(seq_len, num_heads * d_v)
    
    output = heads_out @ W_out
    
    return np.round(output, 4)
