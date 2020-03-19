import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
from . import evaluation


class metricsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        X_val = [self.validation_data[0],self.validation_data[1]]
        y_val = self.validation_data[2]
        y_predict = self.model.predict(x = X_val)
        logs['val_auc'] = evaluation.auc(y_val, y_predict)
        logs['val_rmse'] = evaluation.rmse(y_val, y_predict)
        logs['val_logloss'] = evaluation.logloss(y_val, y_predict)

class metricsEval():
    def __init__(self, model, users, items):
        self.model = model
        self.users = list(set(users))
        self.items = list(set(items))

    def getRecs(self):
        ## create placeholders for user, item, pred
        list_users, list_items, list_preds = [], [], []
        ## get predictions for each user-item pair
        for user in tqdm(self.users):
            user = [user] * len(self.items)
            list_users.extend(user)
            list_items.extend(self.items)
            list_preds.extend(
                self.model.predict(
                    x=[
                        np.array(user),
                        np.array(self.items)
                    ]
                ).flatten()
            )
        ## create a pandas dataframe
        self.all_predictions = pd.DataFrame(data={
            'userID': list_users,
            'itemID': list_items,
            'prediction': list_preds
        })


    def getRankBasedMetrics(self,users_test,items_test,ratings_test,K=10):
        assert K>0, 'Invalid K value for evaluation!!'
        ## format the true test set
        all_test = self.__formatTestDF(users_test,items_test,ratings_test)
        ## calculate the metrics
        recall = evaluation.recall_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )
        precision = evaluation.precision_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )     
        ndcg = evaluation.ndcg_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )
        map = evaluation.map_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )
        return recall,precision,ndcg,map           

    def getOverlapBasedMetrics(self,users_test,items_test,ratings_test):
        y_true = ratings_test
        y_pred = self.model.predict(
            x=[
                np.array(users_test),
                np.array(items_test)
            ]
        ).flatten()
        ## get the metrics
        rmse = evaluation.rmse(y_true, y_pred)
        auc = evaluation.auc(y_true,y_pred)
        logloss = evaluation.logloss(y_true,y_pred)
        return rmse,auc,logloss

    def __formatTestDF(self,users_test,items_test,ratings_test):
        all_test = pd.DataFrame(data={
            'userID':users_test,
            'itemID':items_test,
            'rating':ratings_test
        })
        all_test = all_test[all_test.rating>0].copy().reset_index(drop=True)
        return all_test

