import csv
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import optim, nn

from sklearn import metrics

from tqdm import tqdm

from utils import io_utils


def train_pglcn_iteration(model, args, dataset=None):

    torch.save(
        {
            "model_state": model.state_dict(),
        },
        "log/tmp.cpkt",
    )

    acc_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    prec_list = []
    time_list = []

    print("Repeat 5 fold cross-validation for %s times" % (args.iexp))
    writer = []
    writer.append(["Accuracy", "Recall" , "F1 score", "AUC", "Precision", "Time"])

    if not os.path.exists(("log/" + args.dataset+ "_perform/")):
        os.makedirs(("log/" + args.dataset+ "_perform/"))

    fl_path = "log/%s_perform/%s_%s_lr1_%s_lr2_%f_dropout1_%s_dropout2_%s_dropout3_%s_npc_%s_omic_%s_h_%s_gl_%s_weight_decay_%f" % \
              (args.dataset,args.item,args.method,args.lr1, args.lr2, args.dropout1, args.dropout2, args.dropout3 ,args.npc, args.omic,
               args.hidden_gl, args.hidden_gcn, args.weight_decay)

    save_log = fl_path + ".txt"
    save_csv = fl_path + ".csv"

    for exp in tqdm(range(args.iexp * 5)):
    # for i in tqdm(range(5)):

        best = 10000

        state_dict = torch.load("log/tmp.cpkt")
        model.load_state_dict(state_dict["model_state"])

        test_acc_list = []

        gl_filter_fn = filter(lambda p: p.requires_grad, model.conv_gl.parameters())

        gl_params = list(map(id, model.conv_gl.parameters()))
        gcn_params = filter(lambda p: id(p) not in gl_params,
                            model.parameters())

        gcn_filter_fn = filter(lambda p: p.requires_grad, gcn_params)

        opt1 = optim.Adam(gl_filter_fn, lr=args.lr1, weight_decay=args.weight_decay)
        opt2 = optim.Adam(gcn_filter_fn, lr=args.lr2, weight_decay=args.weight_decay)

        torch.optim.lr_scheduler.StepLR(opt1, 20, gamma=0.9, last_epoch=-1)
        torch.optim.lr_scheduler.StepLR(opt2, 20, gamma=0.9, last_epoch=-1)

        model.train()

        train_mask, val_mask, test_mask = dataset[3][exp]
        X = np.stack(dataset[2])
        train_x = torch.tensor(X[train_mask,], dtype=torch.float32)
        test_x = torch.tensor(X[test_mask,], dtype=torch.float32)
        val_x = torch.tensor(X[val_mask,], dtype=torch.float32)

        labels = dataset[4]
        train_y = torch.tensor(labels[train_mask], dtype=torch.float32)
        test_y = labels[test_mask]
        val_y = torch.tensor(labels[val_mask], dtype=torch.float32)

        logger = []
        if args.gpu:
            train_x = train_x.cuda()
            test_x = test_x.cuda()
            val_x = val_x.cuda()
            train_y = train_y.cuda()
            val_y = val_y.cuda()

        t = time.time()
        for epoch in range(args.epoch):


            model.zero_grad()
            # opt1.zero_grad()
            # opt2.zero_grad()

            # Training step
            logits = model(feat=train_x)
            train_acc, train_loss, train_loss1, train_loss2 = model.loss(logits, train_y)
            # loss1.backward(retain_graph=True)
            # loss2.backward()
            # torch.autograd.set_detect_anomaly(True)
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            opt1.step()
            opt2.step()

            # out
            model.eval()
            with torch.no_grad():
                logit_val = model(feat=val_x, type="predict")
                logit_test = model(feat=test_x, type="predict")

                val_acc, val_loss, val_loss1, val_loss2 = model.loss(logit_val, val_y)

                pred = torch.argmax(logit_test, dim=1).detach().cpu().numpy()
                acc = metrics.accuracy_score(test_y, pred)
                recall = metrics.recall_score(test_y, pred)
                f1_score = metrics.f1_score(test_y, pred)
                auc = metrics.roc_auc_score(test_y, pred)
                prec = metrics.precision_score(test_y, pred)

                logger.append([acc, recall, f1_score, auc, prec])


            cost = val_loss2
            test_acc_list.append(acc)
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss2),
            #       "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_loss2),
            #       "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(acc), "time=",
            #       "{:.5f}".format(time.time() - t))

            if cost < best:
                best_epoch = epoch
                best = cost
                patience = 0

            else:
                patience += 1

            if patience == args.early_stopping:
                # feed_dict_val = construct_feed_dict(features, adj, y_test, test_mask, epoch, placeholders)
                # Smap = sess.run(tf.sparse_tensor_to_dense(model.S), feed_dict=feed_dict_val)
                # sio.savemat("S.mat", {'adjfix': np.array(Smap)})
                break


        # acc, recall, f1_score, auc, prec = logger[-(args.early_stopping + 1)]
        acc, recall, f1_score, auc, prec = logger[- 1]
        time_list.append(time.time() - t)
        acc_list.append(acc)
        recall_list.append(recall)
        f1_list.append(f1_score)
        auc_list.append(auc)
        prec_list.append(prec)

        # print("Optimization Finished!")
        # print("----------------------------------------------")
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss2),
        #       "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_loss2),
        #       "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(acc), "time=",
        #       "{:.5f}".format(time.time() - t))
        # print("acc:", '%.4f' % (acc), "recall:", '%.4f' % (recall), "f1_score:", '%.4f' % (f1_score),
        #       "auc:", '%.4f' % (auc), "prec:", '%.4f' % (prec),)
        # # print("----------------------------------------------")

        # writer.append(["%s,%s,%s,%s,%s" % (str(acc), str(recall), str(f1_score), str(auc), str(prec),)])


    log = pd.DataFrame({
        "Accuracy": acc_list,
        "Recall": recall_list,
        "F1 score":f1_list,
        "AUC": auc_list,
        "Precision": prec_list,        # writer.append([acc,recall,f1_score,auc, prec, time.time() - t])
        # with open(save_log, "w", newline="") as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerows(writer)
        "Time": time_list
    })

    log.to_csv(save_csv, index=False)
    # log.to_csv("log/%s_perform/pglcn.csv" % (args.dataset), index=False)



def train_pglcn(model, args, dataset=None):

    best = 10000

    test_acc_list = []

    gl_filter_fn = filter(lambda p: p.requires_grad, model.conv_gl.parameters())

    gl_params = list(map(id, model.conv_gl.parameters()))
    gcn_params = filter(lambda p: id(p) not in gl_params,
                        model.parameters())

    gcn_filter_fn = filter(lambda p: p.requires_grad, gcn_params)

    opt1 = optim.Adam(gl_filter_fn, lr=args.lr1, weight_decay=args.weight_decay)
    opt2 = optim.Adam(gcn_filter_fn, lr=args.lr2, weight_decay=args.weight_decay)

    torch.optim.lr_scheduler.StepLR(opt1, 20, gamma=0.9, last_epoch=-1)
    torch.optim.lr_scheduler.StepLR(opt2, 20, gamma=0.9, last_epoch=-1)

    model.train()

    train_mask, val_mask, test_mask = dataset[3][0]
    X = np.stack(dataset[2])
    train_x = torch.tensor(X[train_mask,], dtype=torch.float32)
    test_x = torch.tensor(X[test_mask,], dtype=torch.float32)
    val_x = torch.tensor(X[val_mask,], dtype=torch.float32)

    labels = dataset[4]
    train_y = torch.tensor(labels[train_mask], dtype=torch.float32)
    test_y = labels[test_mask]
    val_y = torch.tensor(labels[val_mask], dtype=torch.float32)

    logger = []
    if args.gpu:
        train_x = train_x.cuda()
        test_x = test_x.cuda()
        val_x = val_x.cuda()
        train_y = train_y.cuda()
        val_y = val_y.cuda()

    t = time.time()

    print("Waiting for training...")
    for epoch in range(args.epoch):

        model.zero_grad()
        # opt1.zero_grad()
        # opt2.zero_grad()

        # Training step
        logits = model(feat=train_x)
        train_acc, train_loss, train_loss1, train_loss2 = model.loss(logits, train_y)
        # loss1.backward(retain_graph=True)
        # loss2.backward()
        # torch.autograd.set_detect_anomaly(True)
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        opt1.step()
        opt2.step()

        # out
        model.eval()
        with torch.no_grad():
            logit_val = model(feat=val_x, type="predict")
            logit_test = model(feat=test_x, type="predict")

            val_acc, val_loss, val_loss1, val_loss2 = model.loss(logit_val, val_y)

            pred = torch.argmax(logit_test, dim=1).detach().cpu().numpy()
            acc = metrics.accuracy_score(test_y, pred)
            recall = metrics.recall_score(test_y, pred)
            f1_score = metrics.f1_score(test_y, pred)
            auc = metrics.roc_auc_score(test_y, pred)
            prec = metrics.precision_score(test_y, pred)

            logger.append([acc, recall, f1_score, auc, prec])

        cost = val_loss2
        test_acc_list.append(acc)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss2),
              "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_loss2),
              "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(acc), "time=",
              "{:.5f}".format(time.time() - t))

        if cost < best:
            best_epoch = epoch
            best = cost
            patience = 0

        else:
            patience += 1

        if patience == args.early_stopping:
            # feed_dict_val = construct_feed_dict(features, adj, y_test, test_mask, epoch, placeholders)
            # Smap = sess.run(tf.sparse_tensor_to_dense(model.S), feed_dict=feed_dict_val)
            # sio.savemat("S.mat", {'adjfix': np.array(Smap)})
            break

    print("Optimization Finished!")
    # print("----------------------------------------------")
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss2),
          "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_loss2),
          "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(acc), "time=",
          "{:.5f}".format(time.time() - t))
    print("acc:", '%.4f' % (acc), "recall:", '%.4f' % (recall), "f1_score:", '%.4f' % (f1_score),
          "auc:", '%.4f' % (auc), "prec:", '%.4f' % (prec),)
    # print("----------------------------------------------")

    # writer.append(["%s,%s,%s,%s,%s" % (str(acc), str(recall), str(f1_score), str(auc), str(prec),)])

    sgraph = model.sgraph.detach().cpu().numpy()
    cg_data = {
        "sgraph": sgraph,
        "placeholder": model.placeholder,
        "pred": logits,
        "path": dataset[5]
    }
    io_utils.save_checkpoint(model, [opt1, opt2], args, num_epochs=-1, cg_dict=cg_data)
    # with open(args.dataset +"_writer.pickle", "wb") as f:
    #     pickle.dump(log, f)

    # log extract feature
    raw_feature = X
    extract_feature = model.extract_feature(torch.tensor(X, dtype=torch.float32).cuda())

    raw_feature = raw_feature.reshape(raw_feature.shape[0], -1)
    labels = np.squeeze(dataset[4])
    extract_feature = np.array(extract_feature.cpu().detach())
    if len(extract_feature.shape) > 2:
        extract_feature = extract_feature.reshape(extract_feature.shape[0], -1)

    par_dir = "log/" + args.dataset + "_explain/"

    np.savetxt(par_dir + "raw_feature.csv", raw_feature, delimiter=',')
    np.savetxt(par_dir +  "labels.csv", labels, delimiter=',')
    np.savetxt(par_dir + "extract_feature.csv", extract_feature, delimiter=',')
