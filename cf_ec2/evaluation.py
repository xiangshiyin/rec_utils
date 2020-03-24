
#### version 1

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,log_loss
)

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def mae(y_true,y_pred):
    return mean_absolute_error(y_true,y_pred)

def rsquared(y_true,y_pred):
    return r2_score(y_true,y_pred)

def auc(y_true,y_pred):
    return roc_auc_score(y_true,y_pred)

def logloss(y_true,y_pred):
    return log_loss(y_true,y_pred)

def explainedVar(y_true,y_pred):
    return explained_variance_score(y_true,y_pred)

def mergeTrueAndPred(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_pred
):
    rating_combined = pd.merge(
        rating_true,
        rating_pred.rename(columns={col_pred:col_rating}),
        on=[col_user, col_item],
        suffixes=['_true','_pred']
    )
    return rating_combined[col_rating+'_true'], rating_combined[col_rating+'_pred']

def mergeTrueAndPredWithRank(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_pred,
    k
):
    topK = k
    ## find shared users between pred and actual
    common_users = set(rating_true[col_user].unique()).intersection(
        set(rating_pred[col_user].unique())
    )
    ## clean the pred and actual based on the shared users
    n_users = len(common_users)
    rating_true_common = rating_true[rating_true[col_user].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
    ## get ranked pred output
    topKItems = getTopK(rating_pred_common,col_user,col_pred,topK)
    ## match ranked pred with actual
    df_hit = pd.merge(
        topKItems,
        rating_true_common,
        on=[col_user,col_item]
    )[
        [col_user,col_item,'rnk']
    ]
    ## append actual count
    df_hit_count = pd.merge(
        df_hit.groupby(col_user,as_index=False)[col_user].agg({'hit':'count'}),
        rating_true_common.groupby(col_user,as_index=False)[col_user].agg({'actual':'count'}),
        on=col_user
    )
    return df_hit,df_hit_count,n_users


def getTopK(df,col_user,col_rating,k):
    topKItems = df.groupby(col_user, as_index=False)\
        .apply(lambda items: items.nlargest(k,col_rating))\
            .reset_index(drop=True)
    ## append rank
    topKItems['rnk'] = topKItems.groupby(col_user,sort=False).cumcount()+1
    return topKItems

def precision_at_k(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_pred,
    k
):
    df_hit, df_hit_count, n_users = mergeTrueAndPredWithRank(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_pred=col_pred,
        k=k
    )
    if df_hit.shape[0]==0:
        return 0.0
    return (df_hit_count['hit']/k).sum() / n_users

def recall_at_k(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_pred,
    k
):
    df_hit, df_hit_count, n_users = mergeTrueAndPredWithRank(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_pred=col_pred,
        k=k
    )
    if df_hit.shape[0]==0:
        return 0.0
    return (df_hit_count['hit']/df_hit_count['actual']).sum() / n_users

def ndcg_at_k(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_pred,
    k
):
    df_hit, df_hit_count, n_users = mergeTrueAndPredWithRank(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_pred=col_pred,
        k=k
    )
    if df_hit.shape[0]==0:
        return 0.0

    # calculate discounted gain for hit items
    df_dcg = df_hit.copy()
    # relevance in this case is always 1
    df_dcg["dcg"] = 1 / np.log1p(df_dcg["rnk"])
    # sum up discount gained to get discount cumulative gain
    df_dcg = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})
    # calculate ideal discounted cumulative gain
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=[col_user])
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, k) + 1)))
    )

    # DCG over IDCG is the normalized DCG
    return (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users

def map_at_k(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_pred,
    k
):
    df_hit, df_hit_count, n_users = mergeTrueAndPredWithRank(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_pred=col_pred,
        k=k
    )
    if df_hit.shape[0]==0:
        return 0.0
    df_hit_sorted = df_hit.copy()
    df_hit_sorted['relevance'] = (df_hit_sorted.groupby(col_user).cumcount() + 1)/df_hit_sorted['rnk']
    df_hit_sorted_agg = df_hit_sorted.groupby(col_user).agg({'relevance':'sum'}).reset_index()
    df_combo = pd.merge(
        df_hit_sorted_agg,
        df_hit_count,
        on=col_user
    )
    return (df_combo['relevance']/df_combo['actual']).sum() / n_users