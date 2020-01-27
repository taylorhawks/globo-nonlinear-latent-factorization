# Nonlinear Latent Factorization

## Introduction
This project was inspired by research done at [Condé Nast](https://www.condenast.com/) presented at an event by [Dataiku](https://www.dataiku.com/) on October 23, 2019.

The starting point for the idea of a content-based recommendation system based on Nonlinear Latent Factorization (NLF) is a paper by [Weston et. al. (2013)](https://www.researchgate.net/publication/262245974_Nonlinear_latent_factorization_by_embedding_multiple_user_interests) in which they describe a system of using multiple "interest units" to describe each user and differentiating between them by only using the one providing the best user-item relationship while ignoring the others.

From the paper:
> The key idea of the proposed model is to define T interest vectors per user, where the user part of the model is written as Uˆ which is an m × |U| × T tensor. Hence, we also write Uˆiu ∈ R m as the m-dimensional vector that represents the ith of T possible interests for user u. The item part of the model is the same as in the classical user-item factorization models, and is still denoted as a m×|D| matrix V.

This is a bit abstract still, but makes more sense in the context of content-based recommendations.   The original paper optimizes both User and Item vectors to get the best user-item pairings, in the typical style of a fully collaborative recommendation system.  Instead, this project uses doc2vec-generated item vectors so that the original relative meaning between items-based on their actual content-is not lost.

## Some Math
### Variable Glossary
| name | definition      |   | name | definition      |
|------|-----------------|---|------|-----------------|
| var1 | var1 definition |   | var5 | var5 definition |
| var2 | var2 definition |   | var6 | var6 definition |
| var3 | var3 definition |   | var7 | var7 definition |
| var4 | var4 definition |   | var8 | var8 definition |

### Original Cost Function
<img src = 'img/math/cost_funciton.png'/>
_description_

### Modified Cost Function
<img></img>
_description_

### Gradient
<img></img>
_description_

### Gradient Descent - User Vector Update
<img></img>
_description_

## Project Outline

Skipping the first step (for now), I've chosen a dataset from kaggle with user interactions on the Brazilian news site [Globo](https://www.globo.com/).  They've already vectorized each document with vector length *m* = 250.

### Model 1 - Linear Latent Factorization

#### Pipeline Notes
#### Evaluation

### Model 2 - Nonlinear Latent Factorization
#### Pipeline Notes
#### Evaluation
---


### To-Do:
- Whiteboarding Finished!!
- Code Linear Model
  - Optional but should do: Variable sizes per user (maybe do this at the end)
  - Optional: Use TQDM for GD function
  - Optional: Adaptive learning rate
  - Train/test validation
- Code Nonlinear Model
  - Inherit as much as possible from LinearModel parent class
  - Cluster initialization
  - Max-nonlinearity


---

Skipping doc2vec process for now...

This kaggle dataset has document vectors and readership: https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom

Research:
- http://www.ee.columbia.edu/~ronw/pubs/recsys2013-usermax.pdf

<b> Interests are defined on a user-by-user basis, up to <i>T</i> interests. </b>
  

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
