import os

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

def train_ML(model, args, dataset=None):

    acc_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    prec_list = []

    print("Repeat 5 fold cross-validation for %s times" % (args.iexp))
    for i in tqdm(range(args.iexp*5)):
        # prepare dataset
        train_mask, val_mask, test_mask = dataset[3][i]

        labels = dataset[4]

        X = np.stack([np.concatenate(item)  for item in dataset[2]],axis=0)
        train_x = X[train_mask,]
        train_y = labels[train_mask, ]
        test_x = X[test_mask,]
        test_y = labels[test_mask, ]

        # train model
        model.fit(train_x, np.concatenate(train_y))
        pred = model.predict(test_x)

        # evaluate
        acc = metrics.accuracy_score(test_y, pred)
        recall = metrics.recall_score(test_y, pred)
        f1_score = metrics.f1_score(test_y, pred)
        auc = metrics.roc_auc_score(test_y, pred)
        prec = metrics.precision_score(test_y, pred)

        acc_list.append(acc)
        recall_list.append(recall)
        f1_list.append(f1_score)
        auc_list.append(auc)
        prec_list.append(prec)

    log = pd.DataFrame({
        "Accuracy": acc_list,
        "Recall": recall_list,
        "F1 score":f1_list,
        "AUC": auc_list,
        "Precision": prec_list
    })
    if not os.path.exists(("log/" + args.dataset+ "_perform/")):
        os.makedirs(("log/" + args.dataset+ "_perform/"))

    save_path = "log/" + args.dataset + "_perform/" + args.method + ".csv"
    log.to_csv(save_path, index=False)
