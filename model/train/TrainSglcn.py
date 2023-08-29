import time

import torch
from torch import optim, nn


def train_sglcn(model, args, log=None):

    test_acc_list = []
    best_epoch = 0
    best = 10000

    gl_filter_fn = filter(lambda p : p.requires_grad, model.conv_gl.parameters())

    gl_params = list(map(id, model.conv_gl.parameters()))
    gcn_params = filter(lambda p: id(p) not in gl_params,
                         model.parameters())

    gcn_filter_fn = filter(lambda p : p.requires_grad, gcn_params)

    opt1 = optim.Adam(gl_filter_fn, lr=args.lr1, weight_decay=args.weight_decay)
    opt2 = optim.Adam(gcn_filter_fn, lr=args.lr2, weight_decay=args.weight_decay)
    model.train()

    for epoch in range(args.epoch):

        t = time.time()

        # model.zero_grad()
        opt1.zero_grad()
        opt2.zero_grad()

        # Training step
        train_acc = model()
        loss, loss1, loss2 = model.loss()
        # loss1.backward(retain_graph=True)
        # loss2.backward()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        opt1.step()
        opt2.step()

        # Validation
        cost = None
        acc = None
        test_acc = None

        # out
        model.eval()
        with torch.no_grad():
            val_outs = model.val()
            test_acc = model.test()
        cost = val_outs[0]
        test_acc_list.append(test_acc)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss),
              "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_outs[0]),
              "val_acc=", "{:.5f}".format(val_outs[1]), "test_acc=", "{:.5f}".format(test_acc), "time=",
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
    print("----------------------------------------------")
    print("The finall result:", test_acc_list[-101])
    print("----------------------------------------------")
