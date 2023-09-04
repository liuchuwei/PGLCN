from utils.preprocess import *
import torch
import numpy as np
from model.model_factory import *
import networkx as nx
from params.ML_params import require_ML_params

def obtain_placeholders(args, dataset):

    if args.method == "sglcn":

        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dataset

        # Some preprocessing
        features = preprocess_features(features)
        adj, edge = preprocess_adj(adj)

        adj = np.expand_dims(adj, axis=0)
        features = np.expand_dims(features, axis=0)

        # Create model
        placeholders = {
            'adj': torch.tensor(adj, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(y_train, dtype=torch.float32),
            'labels_mask': torch.tensor(train_mask, dtype=torch.float32),
            'val_labels': torch.tensor(y_val, dtype=torch.float32),
            'test_labels': torch.tensor(y_test, dtype=torch.float32),
            'val_mask': torch.tensor(val_mask, dtype=torch.float32),
            'test_mask': torch.tensor(test_mask, dtype=torch.float32),
            'dropout': args.dropout,
            'num_nodes': torch.tensor(adj.shape[1], dtype=torch.int32),
            'num_features': features.shape[2],  # helper variable for sparse dropout
            'edge': edge
        }

    if args.method == "pglcn":

        adj, features, second, masks_list, labels, Reac_feat = dataset

        # Some preprocessing
        features = preprocess_features(features)
        adj, edge = preprocess_adj(adj)

        adj = np.expand_dims(adj, axis=0)
        features = np.expand_dims(features, axis=0)

        # Create model
        placeholders = {
            'adj': torch.tensor(adj, dtype=torch.float32),
            'features': torch.tensor(features, dtype=torch.float32),
            'dropout': args.dropout,
            'num_nodes': torch.tensor(adj.shape[1], dtype=torch.int32),
            'num_features': features.shape[2],  # helper variable for sparse dropout
            'edge': edge,
            # 'second': [torch.tensor(item, dtype=torch.float32) for item in second],
            'second': torch.tensor(np.stack(second), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),

        }

    return placeholders

def build_explainer(cg_dict, model, args):

    if args.method == "glcn":

        explainer = Explainer(
            model=model,
            adj=cg_dict["sgraph"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            args=args,
            writer=None,
            print_training=True,
            graph_mode=False,
            graph_idx=args.graph_idx,
        )

    elif args.method == "pglcn":
        explainer = Explainer(
            model=model,
            adj=cg_dict["sgraph"],
            feat=cg_dict["placeholder"]["second"],
            label=cg_dict["placeholder"]["labels"],
            pred=cg_dict["pred"],
            train_idx=None,
            args=args,
            writer=None,
            print_training=True,
            graph_mode=False,
            graph_idx=args.graph_idx,
        )

    else:
        explainer = Explainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            args=args,
            writer=None,
            print_training=True,
            graph_mode=False,
            graph_idx=args.graph_idx,
        )

    return explainer

def build_model(args, placeholders=None, dataset=None, **kwargs):

    print("Method:", args.method)

    if args.dataset in ["syn1", "syn2", "syn3", "syn4", "syn5"] and args.method == "gcn":

        if args.command == "train":
            _, labels, _ = dataset
            num_classes = max(labels) + 1

            model = GcnEncoder(
                args.input_dim,
                args.hidden_dim,
                args.output_dim,
                num_classes,
                args.num_gc_layers,
                bn=args.bn,
                args=args
            )

        elif args.command == "explain":
            input_dim = dataset["feat"].shape[2]
            num_classes = dataset["pred"].shape[2]

            model = GcnEncoder(
                input_dim = input_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.output_dim,
                label_dim=num_classes,
                num_layers=args.num_gc_layers,
                bn=args.bn,
                args=args
            )

    if args.dataset in ["syn1", "syn2", "syn3", "syn4", "syn5"] and args.method == "glcn":

        if args.command == "train":

            G, labels, _ = dataset
            adj = np.array(nx.to_numpy_array(G))
            adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
            edge = np.array(np.nonzero(adj_normalized.todense()))
            num_classes = max(labels) + 1

            model = GlcnEncoder(
                args.input_dim,
                args.hidden_dim,
                args.output_dim,
                num_classes,
                args.num_gc_layers,
                bn=args.bn,
                args=args,
                edge=edge
            )


        elif args.command == "explain":
            input_dim = dataset["feat"].shape[2]
            num_classes = dataset["pred"].shape[2]

            adj = np.squeeze(dataset['adj'])
            adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
            edge = np.array(np.nonzero(adj_normalized.todense()))

            model = GlcnEncoder(
                input_dim,
                args.hidden_dim,
                args.output_dim,
                num_classes,
                args.num_gc_layers,
                bn=args.bn,
                args=args,
                edge=edge
            )

    if args.method == "sglcn":
        model = SGLCN(args, placeholders=placeholders)

    if args.method == "pglcn":
        model = PGLCN(args, placeholders=placeholders)

    if args.method in ["sgd", "svc_rbf", "svc_linear", "random_forest",
                             "adaboost", "decision_tree"]:
        params = require_ML_params(args.method)
        model = construct_model(params=params, model_type=args.method)

    return model