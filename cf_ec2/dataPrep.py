import os
import sys
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

DIR_CURRENT = os.path.dirname(os.path.abspath(__file__))
DIR_PARENT = os.path.dirname(DIR_CURRENT)
## customized modules
sys.path.append(DIR_PARENT)

class Data:
    def __init__(
        self,
        train,
        col_user,
        col_item,
        col_rating,
        col_time=None,
        test=None,
        binary=False,
        n_neg=4,
        n_neg_limit=0,
        n_neg_test=100
    ):
        '''
        By default, the input training data is assumed to be pandas dataframe.
        '''
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_time = col_time
        self.train,self.test = self.process(train,test,binary) # indexed train and test sets
        self.n_neg = n_neg
        self.n_neg_limit = n_neg_limit
        self.n_neg_test = n_neg_test
    
    def process(self, train, test, binary):
        '''
        1. Index users and items
        2. Create train and test dataset with only indexed users and items
        '''
        df = train if test is None else train.append(test)
        users = df[self.col_user].unique()
        items = df[self.col_item].unique()
        ## map user and item to IDs
        self.user2id = {value:index for index,value in enumerate(users)}
        self.item2id = {value:index for index,value in enumerate(items)}
        ## map IDs to user and item
        self.id2user = {value:key for key,value in self.user2id.items()}
        self.id2item = {value:key for key,value in self.item2id.items()}
        return self.indexUserItem(train, binary), self.indexUserItem(test, binary)
    
    def indexUserItem(self, df, binary):
        '''
        Index users and items based on the known universe (train+test)
        '''
        if df is None:
            return None
        ## add columns for user and item indicies
        df[self.col_user+'_idx'] = df[self.col_user].map(lambda user: self.user2id[user])
        df[self.col_item+'_idx'] = df[self.col_item].map(lambda item: self.item2id[item])
        ## convert rating to binary value
        if binary:
            df[self.col_rating] = df[self.col_rating].map(lambda rating: float(rating>0))
        ## collect indexed columns for output
        df_indexed = df[[
            self.col_user+'_idx',
            self.col_item+'_idx',
            self.col_rating
        ]].rename(columns={
            self.col_user+'_idx':self.col_user,
            self.col_item+'_idx':self.col_item
        })
        ## remove duplicates
        df_indexed = df_indexed.groupby([
            self.col_user,
            self.col_item,
            self.col_rating
        ]).size().reset_index().drop(columns=0)
        return df_indexed

    def splitTrainTest(self):
        pass

    def prepTrainDNN(self, negSample=True):
        '''
        1. Create positive and negative samples for each users in train
        2. Get lists of users, items, and ratings from train
        '''
        self.itemSet_train = set(self.train[self.col_item].unique()) ## all items in train
        self.interaction_train = self.train.groupby(self.col_user)[self.col_item]\
            .apply(set)\
                .reset_index()\
                    .rename(columns={
                        self.col_item:self.col_item+'_interacted'
                    })
        if not self.n_neg_limit:
            self.interaction_train[self.col_item+'_negative'] = self.interaction_train[
                self.col_item+'_interacted'
            ].map(lambda items: self.itemSet_train-items) ## the actual negative set for each user
        else: # if n_neg_limit is set due to computation resource limit
            self.interaction_train[self.col_item+'_negative'] = [
                random.sample(
                    self.itemSet_train - self.interaction_train.loc[self.interaction_train.user==user, self.col_item+'_interacted'].values[0],
                    self.n_neg_limit
                )
                for user in tqdm(self.interaction_train.user)
            ]

        if negSample:
            self.negativeSampling()
        else:
            ## users, items, ratings all have the same size
            self.users = np.array(self.train[self.col_user])
            self.items = np.array(self.train[self.col_item])
            self.ratings = np.array([
                float(rating)
                for rating in self.train[self.col_rating]
            ])
    
    def prepTestDNN(self, group=False, cleanItem=True):
        '''
        Create the test dataset with negative sampling
        '''
        # t1 = time.time()
        assert self.test is not None, 'No test dataset is assigned!!'
        if cleanItem: ## remove items that are not in train dataset
            self.test = pd.merge(
                self.test,
                self.train.groupby('item').size().reset_index().drop(columns=0),
                on='item'
            )
        interaction_test = self.test.groupby(self.col_user)[self.col_item]\
            .apply(set)\
                .reset_index()\
                    .rename(columns={
                        self.col_item:self.col_item+'_interacted_test'
                    }) ## all items in test
        interaction_test = pd.merge(
            interaction_test,
            self.interaction_train,
            on=self.col_user,
            how='inner'
        ) ## there could be new users showing up in test set, and left join will result in null matches in this case
        # print('Finished initial join with train in {} seconds'.format(time.time()-t1))
        ## generate the negative sample set (based on negative set in training data)
        #### this is a low efficiency code
        # interaction_test[self.col_item+'_negative']=interaction_test.apply(
        #     lambda row: row[self.col_item+'_negative']-row[self.col_item+'_interacted_test'],
        #     axis=1
        # )
        if not self.n_neg_limit: ## without negative sample pool limit
            #### the efficient solution
            for row in interaction_test.itertuples():
                interaction_test.at[row.Index,self.col_item+'_negative'] \
                    = getattr(row,self.col_item+'_negative') - getattr(row,self.col_item+'_interacted_test')
        else: ## with negative sample pool limit
            for row in interaction_test.itertuples():
                interaction_test.at[row.Index,self.col_item+'_negative'] \
                    = set(getattr(row,self.col_item+'_negative')) - getattr(row,self.col_item+'_interacted_test')
        # print('Finished negative sample clean in {} seconds'.format(time.time()-t1))
        ## assign full negative sample set to each record in test dataset
        testPlusNegSample = pd.merge(
            self.test,
            interaction_test[[self.col_user, self.col_item+'_negative']],
            on=self.col_user,
            how='inner'
        )
        # print('Finished negative sample assignment in test data in {} seconds'.format(time.time()-t1))
        ## reduce the negative set by random sampling
        try:
            # testPlusNegSample[self.col_item+'_negative'] = \
            #     testPlusNegSample[self.col_item+'_negative'].map(
            #         lambda negSet: random.sample(negSet, self.n_neg_test)
            #     )
            for row in testPlusNegSample.itertuples():
                testPlusNegSample.at[row.Index, self.col_item+'_negative'] \
                    = random.sample(getattr(row,self.col_item+'_negative'), self.n_neg_test)

        except:
            minNegNum = interaction_test[self.col_item+'_negative'].map(lambda negSet: len(negSet)).min()
            # testPlusNegSample[self.col_item+'_negative'] = \
            #     testPlusNegSample[self.col_item+'_negative'].map(
            #         lambda negSet: random.sample(negSet, minNegNum)
            #     )
            for row in testPlusNegSample.itertuples():
                testPlusNegSample.at[row.Index, self.col_item+'_negative'] \
                    = random.sample(getattr(row,self.col_item+'_negative'), minNegNum)
        # print('Finished negative sampling in test data in {} seconds'.format(time.time()-t1))

        ## flatten the test dataset
        #### generate the test data set (include both positive and negative samples)
        if group == True:
            self.testGrouped = {}

        self.users_test,self.items_test,self.ratings_test = [],[],[]
        for row in testPlusNegSample.itertuples():
            user_tmp = getattr(row,self.col_user)
            item_tmp = getattr(row,self.col_item)
            rating_tmp = getattr(row,self.col_rating)

            self.users_test.append(user_tmp)
            self.items_test.append(item_tmp)
            self.ratings_test.append(float(rating_tmp))
            if group == True:
                if self.testGrouped.get(user_tmp,None)==None:
                    self.testGrouped[user_tmp]={}
                    self.testGrouped[user_tmp]['items'] = [item_tmp]
                    self.testGrouped[user_tmp]['ratings'] = [rating_tmp]
                else:
                    self.testGrouped[user_tmp]['items'].append(item_tmp)
                    self.testGrouped[user_tmp]['ratings'].append(float(rating_tmp))


            for negSample in getattr(row, self.col_item+'_negative'):
                self.users_test.append(user_tmp)
                self.items_test.append(negSample)
                self.ratings_test.append(float(0))
                if group == True:
                    self.testGrouped[user_tmp]['items'].append(negSample)
                    self.testGrouped[user_tmp]['ratings'].append(float(0))             
       
    def negativeSampling(self): ## negative sample the train data set
        '''
        Create the train dataset with negative sampling
        '''
        trainPlusNegSample = pd.merge(
            self.train,
            self.interaction_train[[self.col_user, self.col_item+'_negative']],
            on=self.col_user,
            how='inner'
        )
        try:
            for row in trainPlusNegSample.itertuples():
                trainPlusNegSample.at[row.Index, self.col_item+'_negative'] \
                    = random.sample(getattr(row,self.col_item+'_negative'), self.n_neg)

        except:
            minNegNum = self.interaction_train[self.col_item+'_negative'].map(lambda negSet: len(negSet)).min()
            for row in trainPlusNegSample.itertuples():
                trainPlusNegSample.at[row.Index, self.col_item+'_negative'] \
                    = random.sample(getattr(row,self.col_item+'_negative'), minNegNum)
        #### generate the test data set (include both positive and negative samples)
        self.users,self.items,self.ratings = [],[],[]
        for row in trainPlusNegSample.itertuples():
            self.users.append(getattr(row,self.col_user))
            self.items.append(getattr(row,self.col_item))
            self.ratings.append(float(getattr(row,self.col_rating)))

            for negSample in getattr(row, self.col_item+'_negative'):
                self.users.append(getattr(row,self.col_user))
                self.items.append(negSample)
                self.ratings.append(float(0))
            
        self.users = np.array(self.users)
        self.items = np.array(self.items)
        self.ratings = np.array(self.ratings)
    
    def getBatch_train(self, batchSize, shuffle=True):
        '''
        Create batches of training dataset
        '''
        indices = np.arange(len(self.users))
        if shuffle:
            random.shuffle(indices)
        for i in range(len(indices)//batchSize):
            idx_start = i * batchSize
            idx_end = (i+1) * batchSize
            indices_batch = indices[idx_start:idx_end]
            yield [
                self.users[indices_batch],
                self.items[indices_batch],
                self.ratings[indices_batch]
            ]
