
import numpy as np
import multiprocessing
import heapq
import itertools

# Global variables that are shared across processes
_model = None
_dataset = None
_K = None

## this is just to replicate the original paper's dataset
def getRecMultiUser(dataset, model, K, num_process=1):
    global _model
    global _dataset
    global _K
    _model = model
    _dataset = dataset
    _K = K

    users = set(dataset.users_test)
    # users = list(set(dataset.users_test))[:10]
    if num_process>1: ## multi-processing
        pool = multiprocessing.Pool(processes=num_process)
        results = pool.map(
            getRecSingleUser,
            users,
        )
        pool.close()
        pool.join()
    else: ## single processing
        results = [
            getRecSingleUser(user)
            for user in users
        ]
    ## format result
    results_out = []
    for result in results:
        results_out.extend(result)
    return results_out

def getRecSingleUser(user):
    ## clean items that are not in train dataset
    # items = np.array(dataset.testGrouped[user]['items'])
    items = np.array([
        item
        for item in _dataset.testGrouped[user]['items'] if item in _dataset.items_test
    ])
    users = np.full(
        len(items),
        user,
        dtype='int32'
    )
    ratings_pred = _model.predict(
        x=[users,items]
    ).flatten()
    dict_item_score = {
        item: ratings_pred[idx]
        for idx,item in enumerate(items)
    }
    ## find the topK scored items
    top_k_items = heapq.nlargest(
        _K,
        dict_item_score,
        key=dict_item_score.get
    )
    ## format the output
    #### list of {user,item,rating,rank} value combo
    output = [
        [user, item, dict_item_score[item], idx+1]
        for idx,item in enumerate(top_k_items)
    ]
    return output

def evaluateMultiUser(num_process=4, K=10):
    users = set(_dataset.users_test)
    print(len(users))
    params = list(
        itertools.product(
            users,
            [K]
        )
    )        
    if num_process>1:
        pool = multiprocessing.Pool(processes=num_process)
        result = pool.starmap(
            evaluateSingleUser,
            params
        )
        pool.close()
        pool.join()
        recalls = [r[0] for r in result]
        precisions = [r[1] for r in result]
    else:
        recalls = []
        precisions = []
        for user in users:
            recall, precision = evaluateSingleUser(user,K)
            recalls.append(recall)
            precisions.append(precision)
    return recalls, precisions


def evaluateSingleUser(user, K=10):
    # items = np.array(dataset.testGrouped[user]['items'])
    items = np.array([
        item
        for item in _dataset.testGrouped[user]['items'] if item in _dataset.items_test
    ])        
    users = np.full(
        len(items),
        user,
        dtype='int32'
    )  
    ratings_pred = _model.predict(
        x=[users,items]
    )  
    dict_item_score = {
        item: ratings_pred[idx]
        for idx,item in enumerate(items)
    }
    ## find the topK scored items
    top_k_items = heapq.nlargest(
        K,
        dict_item_score,
        key=dict_item_score.get
    )
    return getHitMetic(items,top_k_items)

def getHitMetic(items_true,items_pred):
    len_overlap = len(
        set(items_true).intersection(
            set(items_pred)
        )
    )
    recall = len_overlap/len(items_true)
    precision = len_overlap/len(items_pred)
    return recall, precision


