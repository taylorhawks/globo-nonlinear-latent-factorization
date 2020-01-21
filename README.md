# globo-nonlinear-latent-factorization

### To-Do:
- Whiteboarding Finished!!
- Fix sampling - maybe need larger sample sizes?
- Code Linear Model
  - vectorize random unseen sampling
- Code Nonlinear Model

Skipping doc2vec process for now...

This kaggle dataset has document vectors and readership: https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom

Research:
- http://www.ee.columbia.edu/~ronw/pubs/recsys2013-usermax.pdf

<b> Interests are defined on a user-by-user basis, up to <i>T</i> interests. </b>
  
  Progress:
    - full rank vs low rank matrix
      https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/dimension-of-the-column-space-or-rank
    - low rank factorization

## User interest partitioning
1. User-by-user clustering
2. Interest unit optimiztion through SGD


#### Possible lead:
H. Keqin, H. Liang, and X. Weiwei. A new effective collaborative filtering algorithm
based on user’s interest partition. In Proceedings of the 2008 International Symposium
on Computer Science and Computational Technology (ISCSCT’08), pages 727–731,
2008

#### Other leads
- https://www.patrickbaudisch.com/interactingwithrecommendersystems/WorkingNotes/RaminYasdiAcquisitionOfUsersInterests.pdf
- https://www.researchgate.net/profile/Doreen_Cheng2/publication/266887747_Situation-aware_User_Interest_Mining_on_Mobile_Handheld_Devices/links/546e31b80cf2b5fc17606f8c.pdf
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.451&rep=rep1&type=pdf


#### THIS COULD BE THE ANSWER
https://www.researchgate.net/publication/221324580_A_New_Effective_Collaborative_Filtering_Algorithm_Based_on_User's_Interest_Partition

#### From the NLF article:
    The key idea of the proposed model is to define T interest vectors per user, where the user part of the model is written as Uˆ which is an m × |U| × T tensor. Hence, we also write Uˆiu ∈ R m as the m-dimensional vector that represents the ith of T possible interests for user u. The item part of the model is the same as in the classical user-item factorization models, and is still denoted as a m×|D| matrix V.
  
  
  
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
