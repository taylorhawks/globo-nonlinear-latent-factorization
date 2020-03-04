import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
from time import time
import matplotlib.pyplot as plt

class NonlinearModel():
    def __init__(self, user_data, item_embeddings, size=8, embeddings_size=250, T=1,norm_U=False):
        self.user_data = user_data
        self.item_embeddings = item_embeddings
        self.len_ = self.item_embeddings.shape[0]
        self.T = T
        self.Vd = NonlinearModel.d.get_embedding_vectors(
            item_embeddings, user_data, size, embeddings_size=embeddings_size
        )
        self.embeddings_size = embeddings_size
        self.U = self.initialize_user_vectors(
            size=size,embeddings_size=embeddings_size
        )
        #normalize user vectors
        if norm_U == True:

            self.U /= np.linalg.norm(self.U, axis=2, keepdims=True)

            #self.U = self.U / np.linalg.norm(self.U, axis=2, keepdims=True)

        self.n_users = int(self.user_data.shape[0]/size)
        self.reset_errors()

    def __add__(self,object):
        '''
        combine models
        concatenates both user data and user vectors
        '''
        pass

    def reset_errors(self):
        self.errors = []

    def initialize_user_vectors(self,size,embeddings_size):
        '''
        sets self.U in parent class __init__
        uses clustering methods defined in the Cluster sub-class
        '''
        starting_vectors = NonlinearModel.d.get_embedding_vectors(
            self.item_embeddings,
            self.user_data,
            size,
            embeddings_size=embeddings_size
        )

        return NonlinearModel.Cluster.k_means(starting_vectors,self.T)
        #return NonlinearModel.Cluster.starting_centroids(starting_vectors,self.T)


    class Cluster():
        @staticmethod
        def starting_centroids(points,T):
            '''
            initialize centroids by randomly picking an item for each user
            '''
            starter_idx = np.random.rand(points.shape[0],T).argsort(axis=-1)[:,0:T,np.newaxis]
            return np.take_along_axis(
                points,
                starter_idx,
                axis=1
            )

        @staticmethod
        def closest_centroid(points, centroids):
            '''
            find the closest centroid to each point using euclidean distance
            it's ok to use euclidean distance for now because clustering just gives a starting point
            the model will be optimized further during gradient descent
            '''
            return np.argmin(
                np.square(
                    np.sum(
                        np.square(points[:,:,np.newaxis] - centroids[:,np.newaxis]),
                        axis=3
                    ),
                ),
                axis=2
            )

        @staticmethod
        def move_centroids(points,closest,k):
            '''
            find new centroid based on cluster center from closest_centroid
            '''


            weights = np.equal(
                np.arange(k)[np.newaxis,:,np.newaxis],
                closest[:,np.newaxis,:]
            )

            weights = np.repeat(weights[:,:,:,np.newaxis],points.shape[2],axis=3)


            points = np.repeat(points[:,np.newaxis],k,axis=1)

            return np.average(
                points,
                weights = weights,
                axis=2
            )

        @staticmethod
        def k_means(starting_vectors,k):
            '''
            Recursive function to optimize kmeans using the functions above,
            starting_centroids, closest_centroid, and move_centroid
            '''
            def iterative_kmeans(starting_vectors,old_centroids, new_centroids,k):
                while True:
                    if np.all(new_centroids == old_centroids):
                        return new_centroids
                    else:
                        old_centroids = new_centroids
                        new_centroids = NonlinearModel.Cluster.move_centroids(
                            starting_vectors,
                            NonlinearModel.Cluster.closest_centroid(
                                starting_vectors,
                                old_centroids
                                ),
                            k,
                        )
                        iterative_kmeans(starting_vectors,old_centroids,new_centroids,k)
            starting_centroids = NonlinearModel.Cluster.starting_centroids(starting_vectors,k)
            return iterative_kmeans(
                starting_vectors,
                np.zeros(starting_centroids.shape),
                starting_centroids,
                k
            )


    #############################
    ###### SEEN AND UNSEEN ######
    #############################

    class d:
        @staticmethod
        def get_embedding_vectors(V_embeddings, df, size, embeddings_size=250):
            return np.hstack(V_embeddings[df.click_article_id.to_list()]).reshape(-1,size,embeddings_size)

    class dbar:
        @staticmethod
        def get_unseen(df, u, size, len_):
            return np.random.choice(
                np.delete(
                    np.arange(len_),
                    df[df.index.get_level_values(0)==u].click_article_id,
                ),
                size=size,
            )

        @staticmethod
        def get_all_unseen(df, size, test_size, len_):
            #semi-vectorized version
            return df.user_id[0::size].map(
                lambda u: NonlinearModel.dbar.get_unseen(df, u, test_size, len_)
            )

        @staticmethod
        def get_embedding_vectors(V_embeddings, df, n_users, size, test_size, embeddings_size=250, len_=36047):

            return np.hstack(
                V_embeddings[NonlinearModel.dbar.get_all_unseen(df,size,test_size,len_).tolist()]
            ).reshape(
                n_users,
                -1,
                embeddings_size
            )

    #############################
    ###### Gradient Descent #####
    #############################

    class gradient():
        @staticmethod
        def argmax_indices(U,Vd):
            '''
            get which interest unit per user, per item, is best for each of their relevant items, d
            output of tensor dot should be 100 x 8 x 3
            output of argmax should be 100 x 8
            '''
            return np.argmax(
                np.tensordot(
                    Vd,
                    U,
                    axes=(2,2)
                )[:,:,0],
                axis=2
            )

        @staticmethod
        def argmax_indices_modified(U,Vd,Vdbar):
            return np.argmin(
                np.tensordot(Vdbar,U,axes=(2,2))[:,:,0] - np.tensordot(Vd,U,axes=(2,2))[:,:,0] ,
                axis=2)

        @staticmethod
        def dJi(Ui,argmax_indices,Vd,Vdbar,T,hinge_param=1):
            '''
            Note that Ui has only the relevant vector (interest unit), while U has all vectors.
            Steps are outlined below.
            '''

            #see if it adds up to more than 0, if it does, it counts toward the cost.
            cond = hinge_param + np.tensordot(Ui,Vdbar,axes=(2,2))[0,0] - np.tensordot(Ui,Vd,axes=(2,2))[0,0] > 0

            #gradient
            g = Vdbar - Vd

            #multiply the calculated gradient by the condition (true and false translate to 1 and 0)
            partial_gradient = g * cond[:,:,np.newaxis]

            #the next part applies the gradient to only the relevant interest unit
            broadcast_gradient=partial_gradient[:,np.newaxis].repeat(T,axis=1)

            #boolean matrix tells which interest unit to update
            boolean_matrix = np.equal(argmax_indices[:,np.newaxis],np.arange(T)[np.newaxis,:,np.newaxis])

            return np.sum(broadcast_gradient * boolean_matrix[:,:,:,np.newaxis], axis=2)


        @staticmethod
        def J(Ui,Vd,Vdbar,hinge_param=1):
            '''
            Total cost calculation. Used if test=True.
            '''
            cost = hinge_param + np.tensordot(Ui,Vdbar,axes=(2,2))[0,0] - np.tensordot(Ui,Vd,axes=(2,2))[0,0]

            # maxed = np.max(
            #     [np.zeros((cost.shape[0],cost.shape[1])), cost], axis = 0
            # )
            #
            # print(maxed.shape)

            return np.sum(
                np.max(
                    [np.zeros((cost.shape[0],cost.shape[1])), cost], axis = 0
                )
            )


    def get_Vdbar_test(self,test_size=128,embeddings_size=250):
        '''
        Get Vdbar for gd validation (used if test=True)
        '''
        self.Vdbar_test = NonlinearModel.dbar.get_embedding_vectors(
                self.item_embeddings,
                self.user_data,
                test_size,
                embeddings_size=self.embeddings_size,
            )



    def gradient_descent_nonlinear(
        self,
        alpha=0.01,
        size=8,
        batch_size=16,
        test_size=128,
        embeddings_size=250,
        test=True,
        use_vdbar_for_interest_unit=False,
        hinge_param=1,
        validation_hinge=1,
        max_iterations=500,
        readj_interval=1,
        gd_algorithm = None
    ):
        '''
        optimize user vectors.

        alpha - learning rate
        size - number of items per user in training data
        batch_size - items to train each user on per iteration
        test_size - if running validation, batch size for calculating objective function
        embeddings_size - number of dimensions for item embeddings
        test - whether to calculate and return validation calculations for objective function (if false, just optimizes)
        hinge_param - L1 regularization parameter for training
        validation_hinge - L1 regularization parameter for validation
        max_iterations - number of rounds of optimization
        readj_interval - how many iterations before assignment of items to their best interest unit is recalculated
        gd_algorithm - if 'rprop', learning rate is step size
        '''
        # self.reset_errors()

        batch_mult = int(batch_size/size)

        if gd_algorithm == 'rprop':
            gd_algorithm = np.sign
        else:
            gd_algorithm = lambda x: x

        for iteration in tqdm(range(max_iterations)):

            #get vdbar for gradient calculation
            Vdbar = NonlinearModel.dbar.get_embedding_vectors(
                self.item_embeddings,
                self.user_data,
                self.n_users,
                size,
                size*batch_mult,
                embeddings_size=self.embeddings_size,
            )

            ##readjustment interval
            if iteration % readj_interval == 0:
                if use_vdbar_for_interest_unit == True:
                    argmax_indices = NonlinearModel.gradient.argmax_indices_modified(self.U,self.Vd,Vdbar)
                else:
                    argmax_indices = NonlinearModel.gradient.argmax_indices(self.U,self.Vd)

            #Ui refers to just the relevant user interest vector. n_users x n_items x m
            Ui = np.take_along_axis(
                self.U,
                argmax_indices[:,:,np.newaxis],
                axis=1
            )



            ### UPDATE ###
            self.U = self.U - alpha / batch_mult * gd_algorithm(NonlinearModel.gradient.dJi(
                Ui.repeat(batch_mult,axis=1), ##
                argmax_indices.repeat(batch_mult,axis=1),
                self.Vd.repeat(batch_mult,axis=1), ##
                Vdbar,
                self.T,
                hinge_param=hinge_param))
            ###############

            if test == True:
                #get Vdbar
                Vdbar_test = NonlinearModel.dbar.get_embedding_vectors(
                    self.item_embeddings,
                    self.user_data,
                    self.n_users,
                    size,
                    test_size,
                    embeddings_size=self.embeddings_size,
                    len_=self.len_
                )


                #add the error at this step to the list of errors so it can be graphed
                #note that this is with a different Vdbar than the one used to calculate gradient
                self.errors.append(
                    NonlinearModel.gradient.J(Ui,self.Vd.repeat(test_size/size,axis=1),Vdbar_test,hinge_param=validation_hinge) / test_size
                )

            #if iteration == max_iterations - 1:
        if test == True:

            plt.figure(figsize=(12,8))
            plt.xlabel('Iterations')
            sns.lineplot(
                x = range(len(self.errors)),
                y = self.errors
            )

        return None

            #iteration += 1

    ################
    ## PREDICTION ##
    ################

    def score_all(self):
        '''
        Score all Ui x V pairs
        Output should be U x T x |items|
        Output should be flattened so that it's a list of T times the number of items |V| for each user
        '''
        self.all_scores = np.tensordot(
            self.U,
            self.item_embeddings,axes=(2,1)
        )

    def filter_scores(self):
        '''
        filter out the items already seen by each user
        '''
        #ones in training set
        already_seen = self.user_data['click_article_id'].to_numpy().reshape((self.n_users,-1))

        #change score to -inf for those ones so they won't be considered.
        np.put_along_axis(self.all_scores,already_seen[:,np.newaxis],-np.inf,axis=2)


    def rank_scores(self):
        start = time()
        self.ranked = np.argsort(
            self.all_scores.reshape(self.U.shape[0],-1),
            axis=1,
            kind='quicksort'
        ) #need to make sure the reshape is doing what I think it's doing
        print(time()-start, 'seconds to sort articles by rank.')

    def get_best_recommendations(self,threshold):
        self.recommended = np.nonzero(self.ranked < threshold)[1].reshape(self.n_users,threshold) % self.len_

    @staticmethod
    def apk(list_):
        i = 1
        sum_ = 0
        for idx,item in enumerate(list_):
            if item == 1:
                sum_ += i/idx
                i +=1
        return(sum_)

    def evaluate(self,test_data,threshold,max_len):
        '''
        max_len is used because there has to be a maximum number of articles per user.
        all others will be the masked value.
        this will provide the range for summing for MAP
        '''
        self.score_all()
        self.filter_scores()
        self.rank_scores()

        self.get_best_recommendations(threshold)
        recommended = self.recommended

        start = time()

        reshaped = np.hstack(test_data.groupby('user_id').apply(
            #fill with -1 for masking
            lambda g: np.pad(g.click_article_id.to_numpy(),(0,max_len-len(g.click_article_id)),'constant',constant_values=-1)
        )).reshape(-1,max_len)


        ground_truth = np.ma.masked_equal(reshaped,-1)[:,:,np.newaxis]

        eval_ = np.sum(np.equal(recommended[:,np.newaxis],ground_truth),axis=1)

        self.mAP = np.mean([NonlinearModel.apk(list_) for list_ in eval_])


        print('Mean Average Precision:',self.mAP)

        print(time() - start, 'seconds to evaluate.')
