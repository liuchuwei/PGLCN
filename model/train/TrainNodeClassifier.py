import pickle
import time

import numpy as np
import torch
from torch import optim, nn

from sklearn import metrics
from utils import gengraph, io_utils


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test


def train_node_classifier(G, labels, model, args, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)
    # scheduler, optimizer = train_utils.build_optimizer(
    #     args, model.parameters(), weight_decay=args.weight_decay
    # )

    filter_fn = filter(lambda p : p.requires_grad, model.parameters())
    optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)

    if args.opt_scheduler == 'none':
        scheduler = None
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)

    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.clip)
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        if writer is not None:
            writer.loss.append(loss)
            writer.epoch.append(epoch)

            # writer.add_scalar("loss/avg_loss", loss, epoch)
            # writer.add_scalars(
            #     "prec",
            #     {"train": result_train["prec"], "test": result_test["prec"]},
            #     epoch,
            # )

            writer.train_prec.append(result_train["prec"])
            writer.test_prec.append(result_test["prec"])

            # writer.add_scalars(
            #     "recall",
            #     {"train": result_train["recall"], "test": result_test["recall"]},
            #     epoch,
            # )

            writer.train_recall.append(result_train["recall"])
            writer.test_recall.append(result_test["recall"])

            # writer.add_scalars(
            #     "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            # )
            #
            writer.train_acc.append(result_train["acc"])
            writer.test_acc.append(result_test["acc"])

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    if args.method == "glcn":
        sgraph = model.sgraph.detach().cpu().numpy()
        cg_data = {
            "adj": data["adj"],
            "sgraph":sgraph,
            "feat": data["feat"],
            "label": data["labels"],
            "pred": ypred.cpu().detach().numpy(),
            "train_idx": train_idx,
        }

    else:
        cg_data = {
            "adj": data["adj"],
            "feat": data["feat"],
            "label": data["labels"],
            "pred": ypred.cpu().detach().numpy(),
            "train_idx": train_idx,
        }
    # import pdb
    # pdb.set_trace()
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    with open(args.dataset +"_writer.pickle", "wb") as f:
        pickle.dump(writer, f)



def train_node_glcn_classifier(G, labels, model, args, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)
    # scheduler, optimizer = train_utils.build_optimizer(
    #     args, model.parameters(), weight_decay=args.weight_decay
    # )

    # filter_fn = filter(lambda p : p.requires_grad, model.parameters())
    # optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    #
    gl_filter_fn = filter(lambda p : p.requires_grad, model.conv_gl.parameters())

    gl_params = list(map(id, model.conv_gl.parameters()))
    gcn_params = filter(lambda p: id(p) not in gl_params,
                         model.parameters())

    gcn_filter_fn = filter(lambda p : p.requires_grad, gcn_params)

    opt1 = optim.Adam(gl_filter_fn, lr=args.lr1, weight_decay=args.weight_decay)
    opt2 = optim.Adam(gcn_filter_fn, lr=args.lr2, weight_decay=args.weight_decay)
    model.train()

    # if args.opt_scheduler == 'none':
    scheduler = None
    # elif args.opt_scheduler == 'step':
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    # elif args.opt_scheduler == 'cos':
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.clip)
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # loss, loss1, loss2 = model.loss()
        # # loss1.backward(retain_graph=True)
        # # loss2.backward()
        # loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        opt1.step()
        opt2.step()
        # optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        if writer is not None:
            writer.loss.append(loss)
            writer.epoch.append(epoch)

            # writer.add_scalar("loss/avg_loss", loss, epoch)
            # writer.add_scalars(
            #     "prec",
            #     {"train": result_train["prec"], "test": result_test["prec"]},
            #     epoch,
            # )

            writer.train_prec.append(result_train["prec"])
            writer.test_prec.append(result_test["prec"])

            # writer.add_scalars(
            #     "recall",
            #     {"train": result_train["recall"], "test": result_test["recall"]},
            #     epoch,
            # )

            writer.train_recall.append(result_train["recall"])
            writer.test_recall.append(result_test["recall"])

            # writer.add_scalars(
            #     "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            # )
            #
            writer.train_acc.append(result_train["acc"])
            writer.test_acc.append(result_test["acc"])

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    if args.method == "glcn":
        sgraph = model.sgraph.detach().cpu().numpy()
        cg_data = {
            "adj": data["adj"],
            "sgraph":sgraph,
            "feat": data["feat"],
            "label": data["labels"],
            "pred": ypred.cpu().detach().numpy(),
            "train_idx": train_idx,
        }

    else:
        cg_data = {
            "adj": data["adj"],
            "feat": data["feat"],
            "label": data["labels"],
            "pred": ypred.cpu().detach().numpy(),
            "train_idx": train_idx,
        }
    # import pdb
    # pdb.set_trace()
    io_utils.save_checkpoint(model, [opt1, opt2], args, num_epochs=-1, cg_dict=cg_data)
    with open(args.dataset +"_writer.pickle", "wb") as f:
        pickle.dump(writer, f)
