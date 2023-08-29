
import os

from matplotlib import pyplot as plt

plt.switch_backend("agg")

import networkx as nx

import numpy as np

from tensorboardX import SummaryWriter

from model.pgexplainer import synthetic_structsim
from utils import featgen
import utils as io_utils

import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold


####################################
#
# Experiment utilities
#
####################################
def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def join_graph(G1, G2, n_pert_edges):
    """ Join two graphs along matching nodes, then perturb the resulting graph.
    Args:
        G1, G2: Networkx graphs to be joined.
        n_pert_edges: number of perturbed edges.
    Returns:
        A new graph, result of merging and perturbing G1 and G2.
    """
    assert n_pert_edges > 0
    F = nx.compose(G1, G2)
    edge_cnt = 0
    while edge_cnt < n_pert_edges:
        node_1 = np.random.choice(G1.nodes())
        node_2 = np.random.choice(G2.nodes())
        F.add_edge(node_1, node_2)
        edge_cnt += 1
    return F


def preprocess_input_graph(G, labels, normalize_adj=False):
    """ Load an existing graph to be converted for the experiments.
    Args:
        G: Networkx graph to be loaded.
        labels: Associated node labels.
        normalize_adj: Should the method return a normalized adjacency matrix.
    Returns:
        A dictionary containing adjacency, node features and labels
    """
    # adj = np.array(nx.to_numpy_matrix(G))
    adj = np.array(nx.to_numpy_array(G))
    if normalize_adj:
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    f = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        f[i, :] = G.nodes[u]["feat"]

    # add batch dim
    adj = np.expand_dims(adj, axis=0)
    f = np.expand_dims(f, axis=0)
    labels = np.expand_dims(labels, axis=0)
    return {"adj": adj, "feat": f, "labels": labels}


####################################
#
# load pathway graphs
#
###################################
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def load_pathway_graph(args, path="data/"):
    """
    load data
    :return:
    """

    # obtain features
    print("loading features...")
    path = path + args.dataset +"/"
    Exp = pd.read_csv(path + args.item + ".csv")
    Reac = Exp.iloc[:,0].tolist()
    Exp = Exp.drop(columns="Unnamed: 0")

    Exp = Exp.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) # min_max normalization

    # obtain pathway dict
    Reac_dict = pd.read_csv("data/react_dict.csv")
    Reac_dict.index = Reac_dict.iloc[:,1]
    Path_name = [Reac_dict.loc[id][2] for id in Reac]
    Reac_feat = pd.DataFrame({"id":Reac, "Path":Path_name})

    # obtain labels
    print("loading labels...")

    labels =  pd.read_csv(path + "Phe.csv")
    labels = pd.get_dummies(labels)
    labels = labels.to_numpy(dtype=np.int32)


    omic = args.omic
    pc = args.npc
    sam = int(Exp.shape[1]/omic/pc)
    mRNA, CNV, MET = np.split(Exp.to_numpy(), omic, axis=1)
    mRNA = np.split(mRNA, sam, axis=1)
    CNV = np.split(CNV, sam, axis=1)
    MET = np.split(MET, sam, axis=1)

    sam_list = []
    for i in range(sam):
        s_mRNA = mRNA[i]
        s_CNV = CNV[i]
        s_MET = MET[i]
        s_mul = np.concatenate([s_mRNA, s_CNV, s_MET], axis=1)
        sam_list.append(s_mul)

    # balance sample
    if args.imbalance:
        from imblearn.over_sampling import SMOTE
        smo = SMOTE(random_state=0)
        tmp = [np.expand_dims(np.concatenate(item, axis=0), axis=0) for item in sam_list]
        tmp = np.concatenate(tmp, axis=0)
        tmp_x, tmp_y = smo.fit_resample(tmp, labels)

        labels = np.expand_dims(tmp_y, axis=1)
        tmp_sam_list = np.array_split(tmp_x, tmp_x.shape[0])

        def fn(item):
            item = np.hsplit(item, indices_or_sections=Exp.shape[0])
            # out = np.concatenate([np.expand_dims(i, axis=0) for i in item], axis=0)
            out = np.concatenate(item, axis=0)
            return out

        tmp_sam_list = [fn(i) for i in tmp_sam_list]

        sam_list = tmp_sam_list

    features = np.concatenate(sam_list, axis=1)
    features = sp.csr_matrix(features)
    second = sam_list

    # obtain adjacent matrix
    print("loading adjacent matrix...")
    adj = pd.read_csv(path + args.item + "adj.csv")
    adj = sp.csr_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    masks_list = []
    print("generating mask...")

    for i in range(args.iexp):

        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=666)

        # generate mask

        id = range(labels.shape[0])
        for train1, test in skf.split(id, labels):

            skf1 = StratifiedKFold(n_splits=5,shuffle=True,random_state=666)
            y1 = labels[train1]
            x1 = [id[i] for i in train1]
            idx_test = [id[i] for i in test]

            for train, val in skf1.split(x1, y1):
                idx_train = [x1[i] for i in train]
                idx_val =  [x1[i] for i in val]
                break

            train_mask = sample_mask(idx_train, labels.shape[0])
            val_mask = sample_mask(idx_val, labels.shape[0])
            test_mask = sample_mask(idx_test, labels.shape[0])

            # y_train = np.zeros(labels.shape)
            # y_val = np.zeros(labels.shape)
            # y_test = np.zeros(labels.shape)
            # y_train[train_mask, :] = labels[train_mask, :]
            # y_val[val_mask, :] = labels[val_mask, :]
            # y_test[test_mask, :] = labels[test_mask, :]

            masks_list.append([train_mask, val_mask, test_mask])

    print("finish...")

    return [adj, features, second, masks_list, labels, Reac_feat]



####################################
#
# load cite graphs
#
###################################
import scipy.io as sio

def load_cite_data(args, path="data/"):

    path = path + args.dataset +"/"

    if args.dataset == "cora":

        # read raw data
        cora_content = pd.read_csv(path + 'cora.content',sep='\t',header=None)
        cora_cites = pd.read_csv(path + 'cora.cites', sep='\t', header=None)

        # paper id to index
        content_idx = list(cora_content.index)
        paper_id = list(cora_content.iloc[:, 0])
        mp = dict(zip(paper_id, content_idx))

        # feature matrix
        features = cora_content.iloc[:, 1:-1]
        features = sp.csr_matrix(features)

        # labels: onehot
        labels = cora_content.iloc[:, -1]
        labels = pd.get_dummies(labels)
        labels = labels.to_numpy(dtype = np.int32)

        # adjacent matrix
        mat_size = cora_content.shape[0]
        adj = np.zeros((mat_size, mat_size))
        for i, j in zip(cora_cites[0], cora_cites[1]):
            x = mp[i]
            y = mp[j]
            adj[x][y] = adj[y][x] = 1
        adj = sp.csr_matrix(adj)

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

    # if dataset_str == "cora":
    #     features = sio.loadmat(path + "feature")
    #     features = features['matrix']
    #     adj = sio.loadmat(path + "adj")
    #     adj = adj['matrix']
    #     labels = sio.loadmat(path + "label")
    #     labels = labels['matrix']
    #     idx_train = range(140)
    #     idx_val = range(200, 500)
    #     idx_test = range(500, 1500)

    elif args.dataset == "citeseer":
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['array'].flatten()
        idx_train = range(120)
        idx_val = range(120, 620)
    else:
        features = sio.loadmat(path + "feature")
        features = features['matrix']
        adj = sio.loadmat(path + "adj")
        adj = adj['matrix']
        labels = sio.loadmat(path + "label")
        labels = labels['matrix']
        idx_test = sio.loadmat(path + "test.mat")
        idx_test = idx_test['matrix']
        idx_train = range(60)
        idx_val = range(200, 500)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



####################################
#
# Generating synthetic graphs
#
###################################
"""gengraph.py

   Generating and manipulaton the synthetic graphs needed for the paper's experiments.
   modify from: https://github.com/RexYing/gnn-model-explainer

"""

def gen_syn1(nb_shapes=80, width_basis=300, feature_generator=None, m=5):
    """ Synthetic Graph #1:

    Start with Barabasi-Albert graph and attach house-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["house"]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def gen_syn2(nb_shapes=100, width_basis=350):
    """ Synthetic Graph #2:

    Start with Barabasi-Albert graph and add node features indicative of a community label.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).

    Returns:
        G                 :  A networkx graph
        label             :  Label of the nodes (determined by role_id and community)
        name              :  A graph identifier
    """
    basis_type = "ba"

    random_mu = [0.0] * 8
    random_sigma = [1.0] * 8

    # Create two grids
    mu_1, sigma_1 = np.array([-1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    mu_2, sigma_2 = np.array([1.0] * 2 + random_mu), np.array([0.5] * 2 + random_sigma)
    feat_gen_G1 = featgen.GaussianFeatureGen(mu=mu_1, sigma=sigma_1)
    feat_gen_G2 = featgen.GaussianFeatureGen(mu=mu_2, sigma=sigma_2)
    G1, role_id1, name = gen_syn1(feature_generator=feat_gen_G1, m=4)
    G2, role_id2, name = gen_syn1(feature_generator=feat_gen_G2, m=4)
    G1_size = G1.number_of_nodes()
    num_roles = max(role_id1) + 1
    role_id2 = [r + num_roles for r in role_id2]
    label = role_id1 + role_id2

    # Edit node ids to avoid collisions on join
    g1_map = {n: i for i, n in enumerate(G1.nodes())}
    G1 = nx.relabel_nodes(G1, g1_map)
    g2_map = {n: i + G1_size for i, n in enumerate(G2.nodes())}
    G2 = nx.relabel_nodes(G2, g2_map)

    # Join
    n_pert_edges = width_basis
    G = join_graph(G1, G2, n_pert_edges)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes) + "_2comm"

    return G, label, name


def gen_syn3(nb_shapes=80, width_basis=300, feature_generator=None, m=5):
    """ Synthetic Graph #3:

    Start with Barabasi-Albert graph and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'grid') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["grid", 3]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, m=5
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name


def gen_syn4(nb_shapes=60, width_basis=8, feature_generator=None, m=4):
    """ Synthetic Graph #4:

    Start with a tree and attach cycle-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'Tree').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["cycle", 6]] * nb_shapes

    fig = plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, plugins = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0
    )
    G = perturb([G], 0.01)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    path = os.path.join("log/syn4_base_h20_o20")
    writer = SummaryWriter(path)
    io_utils.log_graph(writer, G, "graph/full")

    return G, role_id, name


def gen_syn5(nb_shapes=80, width_basis=8, feature_generator=None, m=3):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    basis_type = "tree"
    list_shapes = [["grid", m]] * nb_shapes

    plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0
    )
    G = perturb([G], 0.1)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    path = os.path.join("log/syn5_base_h20_o20")
    writer = SummaryWriter(path)

    return G, role_id, name
