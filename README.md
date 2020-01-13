# globo-nonlinear-latent-factorization

Skipping doc2vec process for now...

This kaggle dataset has document vectors and readership: https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom

Research:
- http://www.ee.columbia.edu/~ronw/pubs/recsys2013-usermax.pdf

To do:
- Figure out how to define discrete user "interests", and thus, user embeddings per interest
  - on a per-user basis?
  - based on clustering algo on all documents?
  
  Progress:
    - full rank vs low rank matrix
    - low rank factorization
    
    The key idea of the proposed model is to define T interest
vectors per user, where the user part of the model is written
as Uˆ which is an m × |U| × T tensor. Hence, we also write
Uˆiu ∈ R
m as the m-dimensional vector that represents the
i
th of T possible interests for user u. The item part of the
model is the same as in the classical user-item factorization
models, and is still denoted as a m×|D| matrix V.
  
  
  
- figure out algorithm #5
  - multi-label SVM?
    - https://stackoverflow.com/questions/49465891/building-an-svm-with-tensorflow


Bucketed Random Projection
Locality-Sensitive Hashing - check out blog post from Uber
SGNS Algorithm
T-SNE Dimensionality Reduction
Latent Dirichlet Allocation (LDA)

Word/Doc 2vec
Distributional meaning!
Cross-entropy loss
Two ways to do word2vec
Skip-grams— sliding window
Continuous bag of words
Two training methods
Hierarchical Softmax
Negative Sampling
(also Naive Softmax)
Two Ways to Doc2Vec
PV-DM
PV-DBOW


Might be helpful: https://livebook.manning.com/book/deep-learning-for-search/chapter-6/v-6/
