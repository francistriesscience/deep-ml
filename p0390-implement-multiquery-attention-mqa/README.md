# Multi-Query Attention (MQA)

LINK: https://www.deep-ml.com/problems/390

## Imagine a Group Project!

Imagine you and your friends are working on a big project in the library.

### The Old Way (Multi-Head Attention)

Every single friend has their **own** copy of the big textbook (Key) and their **own** notebook (Value).

- **Friend 1:** Has a book, a notebook, and asks a question.
- **Friend 2:** Has a book, a notebook, and asks a question.
- **Friend 3:** Has a book, a notebook, and asks a question.

The table is messy! There are too many books!

### The New Way (Multi-Query Attention)

Now, imagine there is just **ONE big textbook** (Key) and **ONE big notebook** (Value) in the middle of the table.

- **Friend 1:** Asks a question (Query) and looks at the shared book.
- **Friend 2:** Asks a question (Query) and looks at the _same_ shared book.
- **Friend 3:** Asks a question (Query) and looks at the _same_ shared book.

The table is clean! Everyone shares the heavy stuff, but they can still ask different questions. This makes everything **faster** and uses **less space**!

---

## How the Solution Works

In standard Multi-Head Attention, each "head" has its own `W_Q`, `W_K`, and `W_V` matrices. This means we have to store $N$ sets of Keys and Values in memory (KV Cache), which is expensive during inference.

**Multi-Query Attention (MQA)** changes this:

- **Heads:** Still have their own unique Queries ($W_Q$).s
- **Keys & Values:** All heads SHARE the same Key ($W_K$) and Value ($W_V$) matrices.

### The Steps:

1.  **Prepare Queries:** We calculate a different Query for each head.
2.  **Prepare Key & Value:** We calculate just **ONE** Key and **ONE** Value for everyone.
3.  **Attention Score:**
    - We compare _every_ head's query against the _single_ shared key.
    - `Scores = Q @ K.T`
4.  **Weighted Sum:**
    - We use those scores to take a weighted sum of the _single_ shared value.
5.  **Combine:** We put all the answers from the heads together.

### The Code:

```python
import numpy as np

def multiquery_attention(X, W_queries, W_key, W_value, W_out):
    # 1. Calculate Queries (One for each head)
    # shapes: (num_heads, seq_len, d_k)
    Q = ...

    # 2. Calculate Shared Key and Value (Just ONE for everyone!)
    # shapes: (seq_len, d_k) and (seq_len, d_v)
    K = X @ W_key
    V = X @ W_value

    # 3. Calculate Scores (Broadcast Key to all Heads)
    # (Heads, Seq, D) @ (Seq, D).T -> (Heads, Seq, Seq)
    scores = Q @ K.T / sqrt(d_k)

    # 4. Apply Softmax to get weights
    weights = softmax(scores)

    # 5. Get Output (Broadcast Value to all Heads)
    # Weights @ V -> Head Outputs
    heads_out = weights @ V

    # 6. Combine and Project
    return heads_out.concat() @ W_out
```
