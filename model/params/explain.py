import argparse
from argparse import ArgumentDefaultsHelpFormatter

from utils import io_utils
from utils.model_utils import build_model, build_explainer
from utils.parser_utils import set_defaults_explain
from utils.train_utils import set_seed


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    # project
    parser.add_argument('--project', dest='project',
                           help='Possible values:'
                                'Dataset_method: stad_pglcn...')

    # set seed
    parser.add_argument('--seed', dest='seed', type=float,
                            help='Seed.')
    # dataset & method
    parser.add_argument('--dataset', dest='dataset',
                           help='Input dataset. Possible values: '
                                'stad, syn1, syn2, syn3, syn4, syn5')
    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: gcn, glcn')

    # io utils
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--logdir", dest="logdir", help="Log directory")
    parser.add_argument("--explainer_suffix", dest="explainer_suffix", help="Suffix of explainer")

    # build model
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=True,
        help="whether to use GPU.",
    )

    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )

    # pglcn
    parser.add_argument('--hidden_gl', dest='hidden_gl', type=int,
                        help='Hidden gl dimension')
    parser.add_argument('--hidden_gcn', dest='hidden_gcn', type=int,
                        help='Hidden gcn dimension')
    parser.add_argument('--dropout1', dest='dropout', type=int,
                        help='Graph learn dropout ratio')
    parser.add_argument('--dropout2', dest='dropout', type=int,
                        help='Graph gcn dropout ratio')
    parser.add_argument('--dropout3', dest='dropout', type=int,
                        help='Dense dropout ratio')
    parser.add_argument('--weight_decay', dest='weight_decay', type=int,
                        help='Weight_decay ratio')
    parser.add_argument('--Placeholders', dest='placeholders', type=bool,
                        help='Placeholders')
    parser.add_argument('--Bias', dest='bias', type=bool,
                        help='bias')
    parser.add_argument('--lr1', dest='lr1', type=float,
                            help='Sparse learning rate.')
    parser.add_argument('--lr2', dest='lr2', type=float,
                            help='Ce learning rate.')
    parser.add_argument('--losslr1', dest='losslr1', type=float,
                            help='Sparse loss weight.')
    parser.add_argument('--losslr2', dest='losslr2', type=float,
                            help='Ce loss weight.')
    parser.add_argument('--omic', dest='omic',
                           help='Number of omic.')
    parser.add_argument('--npc', dest='npc',
                           help='Number of pca component.')

    # explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--opt", dest="opt", type=str, help="Optimizer."
    )
    parser.add_argument('--lr', dest='lr', type=float,
                            help='Learining rate.')
    parser.add_argument(
        "--num_epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, help="Batch size."
    )
    parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')

    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')

    return parser

def main(args):

    # set default parser
    set_defaults_explain(args)

    # set seed
    set_seed(args.seed)

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(args)
    cg_dict = ckpt["cg"] # get computation graph

    # build model
    if args.method == "pglcn":
        placeholder = cg_dict["placeholder"]
        model = build_model(args=args, placeholders=placeholder)
    else:
        model = build_model(args=args, dataset=cg_dict)

    # load state_dict
    if args.gpu:
        model = model.cuda()
    model.load_state_dict(ckpt["model_state"])

    # Create explainer
    explainer = build_explainer(cg_dict, model, args)

    # Explain
    if args.method == "pglcn":
        masked_adj = explainer.explain_path_stats(
            0, args
        )
        # save mask
        import numpy as np

        feat_mask = [item[1].detach().cpu().tolist() for item in masked_adj]
        feat_mask = np.array(feat_mask)
        adj_mask = [item[2] for item in masked_adj]
        adj_mask = np.array(adj_mask)
        np.save("log/stad_explain/feat_mask.npy", feat_mask)
        np.save("log/stad_explain/adj_mask.npy", adj_mask)


        # visualization
        ## denoise_graph
        for id in range(adj_mask.shape[0]):
            adj = adj_mask[id]
            out_id = id*25
            import networkx as nx
            num_nodes = adj.shape[-1]
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.nodes[0]["self"] = 1

            feat = None
            if feat is not None:
                for node in G.nodes():
                    G.nodes[node]["feat"] = feat[node]

            label = range(num_nodes)
            feat = cg_dict['path']
            id = feat['id'].tolist()
            path = feat['Path'].tolist()

            for node in G.nodes():
                G.nodes[node]["label"] = label[node]
                G.nodes[node]["id"] = id[node]
                G.nodes[node]["path"] = path[node]

            threshold = 0.1
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
            max_component = True

            if max_component:
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            else:
                # remove zero degree nodes
                G.remove_nodes_from(list(nx.isolates(G)))

            ## log_graph
            from matplotlib import pyplot as plt
            Gc = G
            cmap = plt.get_cmap("Set1")
            plt.switch_backend("agg")
            fig_size = (4, 3)
            dpi=300
            fig = plt.figure(figsize=fig_size, dpi=dpi)

            node_colors = []
            # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in Gc.edges.data('weight', default=1)]
            edge_colors = [w for (u, v, w) in Gc.edges.data("weight", default=1)]

            # maximum value for node color
            vmax = 8
            nodecolor = "label"
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
            identify_self = True
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

            label_node_feat = False
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
            edge_vmax = None
            import statistics

            if edge_vmax is None:
                edge_vmax = statistics.median_high(
                    [d for (u, v, d) in Gc.edges(data="weight", default=1)]
                )
            min_color = min([d for (u, v, d) in Gc.edges(data="weight", default=1)])
            # color range: gray to black
            edge_vmin = 2 * min_color - edge_vmax




            if out_id < 100:
                node_labels = nx.get_node_attributes(G, 'label')
                nx.draw_networkx_labels(Gc, pos=pos_layout, labels=node_labels, font_size=3)
            if out_id in [100, 125]:
                node_labels = nx.get_node_attributes(G, 'label')
                nx.draw_networkx_labels(Gc, pos=pos_layout, labels=node_labels, font_size=3)
            elif out_id in [150, 175]:
                node_labels = nx.get_node_attributes(G, 'path')
                nx.draw_networkx_labels(Gc, pos=pos_layout, labels=node_labels, font_size=4)
            elif out_id == 200:
                node_labels = nx.get_node_attributes(G, 'path')
                nx.draw_networkx_labels(Gc, pos=pos_layout, labels=node_labels, font_size=5)


            nx.draw(
                Gc,
                pos=pos_layout,
                with_labels=False,
                font_size=4,
                labels=feat_labels,
                # node_color=node_colors,
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


            fig.axes[0].xaxis.set_visible(False)
            fig.canvas.draw()

            plt.savefig("log/stad_explain/explain_%s.pdf" % (out_id), format="pdf")
            plt.close()


    elif args.dataset == "syn1":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(400, 700, 5), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [300], args
        )

    elif args.dataset == "syn2":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(400, 700, 5), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [350], args
        )

    elif args.dataset == "syn3":


        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(301, 1021, 9), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [301], args
        )

    elif args.dataset == "syn4":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(511, 871, 6), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [511], args
        )

    elif args.dataset == "syn5":

        # masked_adj = explainer.explain_nodes_gnn_stats(
        #     range(512, 1232, 9), args
        # )
        masked_adj = explainer.explain_nodes_gnn_stats(
            [548], args
        )
