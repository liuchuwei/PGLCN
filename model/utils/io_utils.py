import os
import statistics

import networkx as nx
import numpy as np
import tensorboardX
import torch
from matplotlib import pyplot as plt

from utils import gengraph, featgen
# dataset: stad, coad, ucec, syn1, syn2, syn3, syn4, syn5, citeseer, cora
def obtain_dataset(args):

    if args.dataset == "syn1":
        G, labels, name = gengraph.gen_syn1(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset == "syn2":
        G, labels, name = gengraph.gen_syn2()
        return G, labels, name

    if args.dataset == "syn3":
        G, labels, name = gengraph.gen_syn3(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset == "syn4":
        G, labels, name = gengraph.gen_syn4(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset == "syn5":
        G, labels, name = gengraph.gen_syn5(
            feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        )
        return G, labels, name

    if args.dataset in ["stad", "coad", "ucec"]:
        adj, features, second, masks_list, labels, Reac_feat = gengraph.load_pathway_graph(args)
        return adj, features, second, masks_list, labels, Reac_feat

    if args.dataset == "cora":
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = gengraph.load_cite_data(args)
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    if args.dataset == "citeseer":
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = gengraph.load_cite_data(args)
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def gen_prefix(args):
    '''Generate label prefix for a graph model.
    '''
    # if args.bmname is not None:
    #     name = args.bmname
    # else:
    name = args.dataset
    name += "_" + args.method

    # name += "_h" + str(args.hidden_dim) + "_o" + str(args.output_dim)
    # if not args.bias:
    #     name += "_nobias"
    # if len(args.name_suffix) > 0:
    #     name += "_" + args.name_suffix
    return name



def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    filename = os.path.join(save_dir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + ".pth.tar"


def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.

    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """

    filename = args.dataset + "_" + args.method + "_pretrain_for_explain." + "pth.tar"
    # create parent directory
    par_dir = "log/" + args.dataset + "_explain"

    if not os.path.exists(par_dir):
        os.makedirs(par_dir)

    filename = par_dir + "/" + filename
    if isinstance(optimizer, list):
        torch.save(
            {
                "epoch": num_epochs,
                "model_type": args.method,
                "optimizer": optimizer,
                "model_state": model.state_dict(),
                "optimizer_state": [i.state_dict() for i in optimizer],
                "cg": cg_dict,
            },
            filename,
        )

    else:
        torch.save(
            {
                "epoch": num_epochs,
                "model_type": args.method,
                "optimizer": optimizer,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cg": cg_dict,
            },
            filename,
        )



def load_ckpt(args, isbest=False):
    '''Load a pre-trained pytorch model from checkpoint.
    '''
    print("loading model")
    # filename = create_filename(args.ckptdir, args, isbest)
    filename = args.ckptdir
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python run.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt


def gen_explainer_prefix(args):
    '''Generate label prefix for a graph explainer model.
    '''
    name = gen_prefix(args) + "_explain"
    if len(args.explainer_suffix) > 0:
        name += "_" + args.explainer_suffix
    return name



def denoise_graph(adj, node_idx, feat=None, label=None, threshold=None, threshold_num=None, max_component=True):
    """Cleaning a graph by thresholding its node values.

    Args:
        - adj               :  Adjacency matrix.
        - node_idx          :  Index of node to highlight (TODO ?)
        - feat              :  An array of node features.
        - label             :  A list of node labels.
        - threshold         :  The weight threshold.
        - theshold_num      :  The maximum number of nodes to threshold.
        - max_component     :  TODO
    """
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.nodes[node_idx]["self"] = 1
    if feat is not None:
        for node in G.nodes():
            G.nodes[node]["feat"] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.nodes[node]["label"] = label[node]

    if threshold_num is not None:
        # this is for symmetric graphs: edges are repeated twice in adj
        adj_threshold_num = threshold_num * 2
        #adj += np.random.rand(adj.shape[0], adj.shape[1]) * 1e-4
        neigh_size = len(adj[adj > 0])
        threshold_num = min(neigh_size, adj_threshold_num)
        threshold = np.sort(adj[adj > 0])[-threshold_num]

    if threshold is not None:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] >= threshold
        ]
    else:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] > 1e-6
        ]
    G.add_weighted_edges_from(weighted_edge_list)
    if max_component:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    else:
        # remove zero degree nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    return G



# TODO: unify log_graph and log_graph2
def log_graph(
    writer,
    Gc,
    name,
    identify_self=True,
    nodecolor="label",
    epoch=0,
    fig_size=(4, 3),
    dpi=300,
    label_node_feat=False,
    edge_vmax=None,
    args=None,
):
    """
    Args:
        nodecolor: the color of node, can be determined by 'label', or 'feat'. For feat, it needs to
            be one-hot'
    """
    cmap = plt.get_cmap("Set1")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_colors = []
    # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in Gc.edges.data('weight', default=1)]
    edge_colors = [w for (u, v, w) in Gc.edges.data("weight", default=1)]

    # maximum value for node color
    vmax = 8
    for i in Gc.nodes():
        if nodecolor == "feat" and "feat" in Gc.nodes[i]:
            num_classes = Gc.nodes[i]["feat"].size()[0]
            if num_classes >= 10:
                cmap = plt.get_cmap("tab20")
                vmax = 19
            elif num_classes >= 8:
                cmap = plt.get_cmap("tab10")
                vmax = 9
            break

    feat_labels = {}
    for i in Gc.nodes():
        if identify_self and "self" in Gc.nodes[i]:
            node_colors.append(0)
        elif nodecolor == "label" and "label" in Gc.nodes[i]:
            node_colors.append(Gc.nodes[i]["label"] + 1)
        elif nodecolor == "feat" and "feat" in Gc.nodes[i]:
            # print(Gc.nodes[i]['feat'])
            feat = Gc.nodes[i]["feat"].detach().numpy()
            # idx with pos val in 1D array
            feat_class = 0
            for j in range(len(feat)):
                if feat[j] == 1:
                    feat_class = j
                    break
            node_colors.append(feat_class)
            feat_labels[i] = feat_class
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels = None

    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if Gc.number_of_nodes() == 0:
        raise Exception("empty graph")
    if Gc.number_of_edges() == 0:
        raise Exception("empty edge")
    # remove_nodes = []
    # for u in Gc.nodes():
    #    if Gc
    pos_layout = nx.kamada_kawai_layout(Gc, weight=None)
    # pos_layout = nx.spring_layout(Gc, weight=None)

    weights = [d for (u, v, d) in Gc.edges(data="weight", default=1)]
    if edge_vmax is None:
        edge_vmax = statistics.median_high(
            [d for (u, v, d) in Gc.edges(data="weight", default=1)]
        )
    min_color = min([d for (u, v, d) in Gc.edges(data="weight", default=1)])
    # color range: gray to black
    edge_vmin = 2 * min_color - edge_vmax

    nx.draw(
        Gc,
        pos=pos_layout,
        with_labels=False,
        font_size=4,
        labels=feat_labels,
        node_color=node_colors,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("Greys"),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=1.0,
        node_size=50,
        alpha=0.8,
    )

    # if args.method == "pglcn":
    #     node_labels = Gc.nodes.data
    #     nx.draw_networkx_labels(Gc, pos=pos_layout, labels=node_labels, font_size=20)

    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    logdir = "log" if not hasattr(args, "logdir") or not args.logdir else str(args.logdir)
    if nodecolor != "feat":
        name += gen_explainer_prefix(args)
    name = args.method + "_" + name
    dataset_dir = args.dataset + "_explain"
    save_path = os.path.join(logdir, dataset_dir, name  + "_" + str(epoch) + ".pdf")
    print(logdir + "/" + name + gen_explainer_prefix(args) + "_" + str(epoch) + ".pdf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf")

    # img = tensorboardX.utils.figure_to_image(fig)
    # writer.add_image(name, img, epoch)