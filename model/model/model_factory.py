import os
import time

import math
import networkx as nx
import numpy as np
import tensorboardX
import torch.nn
from matplotlib import pyplot as plt
from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Ridge, ElasticNet, Lasso, SGDClassifier, RidgeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


from torch.nn import init
from model.layers import *
from model.metrics import *
from utils import graph_utils, io_utils, train_utils


####################################
#
# typical machine learning  models
#
####################################

def construct_model(params, model_type):
    p = params

    if model_type == 'sgd':
        model = SGDClassifier(**p)

    if model_type == 'svc_rbf':
        model = svm.SVC(max_iter=5000, **p)

    if model_type == 'svc_linear':
        model = svm.SVC(max_iter=5000, **p)

    if model_type == 'random_forest':
        model = RandomForestClassifier(**p)

    if model_type == 'adaboost':
        model = AdaBoostClassifier(**p)

    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**p)


    return model


####################################
#
# Deep learning models
#
####################################

class PGLCN(nn.Module):

    def __init__(self, proargs, placeholders, *args, **kwargs):

        super(PGLCN, self).__init__(*args, **kwargs)
        self.args = proargs
        self.placeholder = placeholders

        # input
        if proargs.gpu:
            self.input =  placeholders['features'].cuda()
            self.second = placeholders['second'].cuda()
        else:
            self.input = placeholders['features']
            self.second = placeholders['second']

        # build layers
        self.input_dim = placeholders['features'].shape[2]
        self.hidden_gl = proargs.hidden_gl
        self.omic = proargs.omic * proargs.npc
        self.hidden_gcn = proargs.hidden_gcn
        self.output_dim =  placeholders['labels'].shape[1] + 1
        self.bias = proargs.bias
        self.gpu = proargs.gpu
        self.dropout = proargs.dropout
        self.edge = placeholders['edge']

        self.conv_gl, self.conv_first, self.conv_last = self.build_gl_conv_layers(
            self.input_dim ,
            self.hidden_gl ,
            self.omic,
            self.hidden_gcn
        )
        self.pred_model = self.build_pred_layers()
        if self.gpu:
            self.pred_model = self.pred_model.cuda()

        for m in self.modules():
            if isinstance(m, SparseGraphLearn):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

                if m.a is not None:
                    init.xavier_uniform_(m.a.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))

            if isinstance(m, GraphConvSlice):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
                    # init.xavier_uniform_(m.bias.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))

    def forward(self, feat=None, adj=None, type = "train"):

        if type == "train":
            _, self.sgraph = self.conv_gl(self.input)
            nn = self.apply_bn(feat)
            h0 = self.conv_first(nn, self.sgraph)
            h0 = self.apply_bn(h0)
            # h1 = self.conv_last(h0, self.sgraph)
            h1 = self.conv_last(h0, self.sgraph)
            h1 = self.apply_bn(h1)
            h2 = torch.cat((h0, h1), dim=2)
            h2 = torch.flatten(h2, start_dim=1)
            # h = nn.Dropout(p=self.args.dropout)(h)
            pred = self.pred_model(h2)
            # out = torch.nn.Softmax(dim=0)(pred)
        elif type=="predict":
            _, self.sgraph = self.conv_gl(self.input)
            nn = self.apply_bn(feat)
            h0 = self.conv_first(nn, self.sgraph)
            h0 = self.apply_bn(h0)
            # h1 = self.conv_last(h0, self.sgraph)
            h1 = self.conv_last(h0, self.sgraph)
            h1 = self.apply_bn(h1)
            h2 = torch.cat((h0, h1), dim=2)
            h2 = torch.flatten(h2, start_dim=1)
            # h = nn.Dropout(p=self.args.dropout)(h)
            pred = self.pred_model(h2)
            # out = torch.nn.Softmax(dim=0)(pred)

        else:
            _, self.sgraph = self.conv_gl(self.input)
            nn = self.apply_bn(feat)
            h0 = self.conv_first(nn, self.sgraph)
            h0 = self.apply_bn(h0)
            # h1 = self.conv_last(h0, self.sgraph)
            h1 = self.conv_last(h0, self.sgraph)
            h1 = self.apply_bn(h1)
            h2 = torch.cat((h0, h1), dim=2)
            h2 = torch.flatten(h2, start_dim=1)
            # h = nn.Dropout(p=self.args.dropout)(h)
            pred = self.pred_model(h2)
            # out = torch.nn.Softmax(dim=0)(pred)

        return pred

    def extract_feature(self, x):
        _, self.sgraph = self.conv_gl(self.input)
        nn = self.apply_bn(x)
        h0 = self.conv_first(nn, self.sgraph)
        h0 = self.apply_bn(h0)
        # h1 = self.conv_last(h0, self.sgraph)
        h1 = self.conv_last(h0, self.sgraph)
        h1 = self.apply_bn(h1)
        h2 = torch.cat((h0, h1), dim=2)
        h2 = torch.flatten(h2, start_dim=1)
        pred = self.pred_model(h2)

        return pred

    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def pglcn_forward(self):
        _, self.sgraph = self.conv_gl(self.input)
        embeding = self.conv(self.second, self.sgraph)
        return embeding

    def build_gl_conv_layers(self,
            input_dim ,
            hidden_gl ,
            omic,
            hidden_gcn,
            normalize=False):

        self.conv_gl = SparseGraphLearn(
            input_dim=input_dim,
            output_dim=hidden_gl,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            act=torch.nn.ReLU(),
            dropout=self.args.dropout1,
            edge = self.edge)

        self.conv_first = GraphConvSlice(
            input_dim=omic,
            output_dim=hidden_gcn,
            normalize_embedding=normalize,
            bias=False,
            gpu=self.gpu,
            act=torch.nn.ReLU(),
            dropout=self.args.dropout2,
        )

        self.conv_last = GraphConvSlice(
            input_dim=hidden_gcn,
            output_dim=self.output_dim,
            normalize_embedding=normalize,
            bias=False,
            gpu=self.gpu,
            act=lambda x: x,
            dropout=self.args.dropout2,
        )

        return self.conv_gl, self.conv_first, self.conv_last

    def build_pred_layers(self):

        pred_layers = []
        drop_rate = self.args.dropout3
        input_dim = self.input.size()[1]*(self.output_dim+self.hidden_gcn)
        # pred_layers.append(nn.Linear(input_dim, 4096))
        # pred_layers.append(torch.nn.ReLU())
        # pred_layers.append(nn.Dropout(p=drop_rate))
        # pred_layers.append(nn.Linear(4096, 2048))
        # pred_layers.append(torch.nn.ReLU())
        # pred_layers.append(nn.Dropout(p=drop_rate))
        # pred_layers.append(nn.Linear(2048, 1024))
        # pred_layers.append(torch.nn.ReLU())
        # pred_layers.append(nn.Dropout(p=drop_rate))
        # pred_layers.append(nn.Linear(1024, 512))
        pred_layers.append(torch.nn.ReLU())
        pred_layers.append(nn.Dropout(p=drop_rate))
        pred_layers.append(nn.Linear(input_dim, self.output_dim))
        pred_layers.append(torch.nn.Identity())
        pred_model = nn.Sequential(*pred_layers)

        return pred_model

    def loss(self, logits, labels):

        # mask = labels_mask.clone()
        D = torch.diag(torch.ones(self.placeholder["num_nodes"])) * -1
        D = (D + self.sgraph[0, :, :].cpu()) * -1
        # D = torch.matmul(torch.transpose(self.x[0, :, :]), D)
        # D = torch.matmul(self.x[0, :, :].T.cpu(), D)
        D = torch.matmul(self.input[0, :, :].T.cpu(), D)
        # loss1 = torch.trace(torch.matmul(D, self.x[0, :, :].cpu())) * self.losslr1
        loss1 = torch.trace(torch.matmul(D, self.input[0, :, :].cpu())) * self.args.losslr1
        loss1 -= torch.trace(torch.matmul(self.sgraph[0, :, :].cpu().T, self.sgraph[0, :, :].cpu())) * self.args.losslr2

        # loss2 = masked_softmax_cross_entropy(logits, labels,
        #                                      mask)
        # loss2 = masked_cross_entropy(logits, labels)
        # loss2 = logit_cross_entropy(logits, labels)
        loss2 = logit_BCE(logits, labels)
        # loss2 = softmax_cross_entropy(logits, labels)
        loss = loss1 + loss2

        # acc = masked_accuracy(logits, labels,
        #                       mask)
        acc = logit_accuracy(logits, labels)

        return acc, loss, loss1, loss2

    def _val(self, logits, labels, labels_mask):

        mask = labels_mask.clone()
        D = torch.diag(torch.ones(self.placeholder["num_nodes"])) * -1
        D = (D + self.sgraph[0, :, :].cpu()) * -1
        # D = torch.matmul(torch.transpose(self.x[0, :, :]), D)
        # D = torch.matmul(self.x[0, :, :].T.cpu(), D)
        D = torch.matmul(self.input[0, :, :].T.cpu(), D)
        # loss1 = torch.trace(torch.matmul(D, self.x[0, :, :].cpu())) * self.losslr1
        loss1 = torch.trace(torch.matmul(D, self.input[0, :, :].cpu())) * self.args.losslr1
        loss1 -= torch.trace(torch.matmul(self.sgraph[0, :, :].cpu().T, self.sgraph[0, :, :].cpu())) * self.args.losslr2

        loss2 = masked_cross_entropy(logits, labels,
                                             mask)
        loss = loss1 + loss2

        acc = masked_accuracy(logits, labels,
                              mask)

        return [loss, acc]

    def _test(self, logits, labels, labels_mask):
        mask = labels_mask.clone()
        acc = masked_accuracy(logits, labels,
                              mask)
        return acc


class SGLCN(nn.Module):

    """
    modify from: https://github.com/jiangboahu/GLCN-tf/blob/master/glcn/models.py
    """

    def __init__(self, args, placeholders, **kwargs):
        super(SGLCN, self).__init__()

        # dropout
        self.dropout = args.dropout

        # gpu
        self.gpu = args.gpu

        # bias
        self.bias = args.bias

        # edge
        self.edge = placeholders['edge']

        # input
        if self.gpu :
            self.inputs = placeholders['features'].cuda()
            self.adj = placeholders['adj'].cuda()

            self.labels = placeholders['labels'].cuda()
            self.labels_mask = placeholders['labels_mask'].cuda()

            self.val_labels = placeholders['val_labels'].cuda()
            self.val_mask = placeholders['val_mask'].cuda()

            self.test_labels = placeholders['test_labels'].cuda()
            self.test_mask = placeholders['test_mask'].cuda()
        else:
            self.inputs = placeholders['features']
            self.adj = placeholders['adj']

            self.labels = placeholders['labels']

            self.val_labels = placeholders['val_labels']

            self.test_labels = placeholders['test_labels']
            self.labels_mask = placeholders['labels_mask']
            self.val_mask = placeholders['val_mask']
            self.test_mask = placeholders['test_mask']

        # layers
        self.args = args
        self.placeholders = placeholders
        self.input_dim = placeholders['num_features']
        self.hidden_gl = args.hidden_gl
        self.hidden_gcn = args.hidden_gcn
        self.output_dim = placeholders['labels'].size()[1]

        self.conv_gl, self.conv_first, self.conv_last = self.build_SGLCN_layers(
            self.input_dim ,
            self.hidden_gl ,
            self.hidden_gcn ,
            self.output_dim
        )

        # loss
        self.num_nodes = placeholders['num_nodes']
        self.losslr1 = args.losslr1
        self.losslr2 = args.losslr2

        for m in self.modules():
            if isinstance(m, SparseGraphLearn):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

                if m.a is not None:
                    init.xavier_uniform_(m.a.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))

            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
                    # init.xavier_uniform_(m.bias.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))

        # for m in self.modules():
        #     if isinstance(m, SparseGraphLearn) or isinstance(m, GraphConv):
        #         init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        #         # init.xavier_normal(m.weight.data)
        #
        #     if m.bias is not None:
        #         init.constant_(m.bias.data, 0.0)
        #         # init.xavier_uniform_(m.bias.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))
        #
        #     if m.a is not None:
        #         init.xavier_uniform_(m.a.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))
        #
        #     # if m.bias is not None:
        #     #     # init.constant_(m.bias.data, 0.0)
        #     #     # init.xavier_normal(m.bias.data.unsqueeze(0))
        #     #     init.xavier_uniform_(m.bias.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))

    def build_SGLCN_layers(self,
            input_dim ,
            hidden_gl ,
            hidden_gcn ,
            output_dim ,
            normalize=False):

        self.conv_gl = SparseGraphLearn(
            input_dim=input_dim,
            output_dim=hidden_gl,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            act=torch.nn.ReLU(),
            dropout=self.dropout,
            edge = self.edge)

        self.conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_gcn,
            normalize_embedding=normalize,
            bias=False,
            gpu=self.gpu,
            act=torch.nn.ReLU(),
            dropout=self.dropout,
        )

        self.conv_last = GraphConv(
            input_dim=hidden_gcn,
            output_dim=output_dim,
            normalize_embedding=normalize,
            bias=False,
            act=lambda x: x,
            dropout=self.dropout)

        return self.conv_gl, self.conv_first, self.conv_last

    def forward(self):
        self.x, self.sgraph = self.conv_gl(self.inputs)

        h1, _ = self.conv_first(self.inputs, self.sgraph)
        h2, _ = self.conv_last(h1, self.sgraph)

        self.outputs = torch.softmax(h2[0, :, :], dim=1)

        acc = masked_accuracy(self.outputs, self.labels,
                              self.labels_mask)
        return acc

    def loss(self):
        # D = torch.matrix_diag(torch.ones(self.num_nodes)) * -1
        D = torch.diag(torch.ones(self.num_nodes)) * -1
        D = (D + self.sgraph[0, :, :].cpu()) * -1
        # D = torch.matmul(torch.transpose(self.x[0, :, :]), D)
        # D = torch.matmul(self.x[0, :, :].T.cpu(), D)
        D = torch.matmul(self.inputs[0, :, :].T.cpu(), D)
        # loss1 = torch.trace(torch.matmul(D, self.x[0, :, :].cpu())) * self.losslr1
        loss1 = torch.trace(torch.matmul(D, self.inputs[0, :, :].cpu())) * self.losslr1
        loss1 -= torch.trace(torch.matmul(self.sgraph[0, :, :].cpu().T, self.sgraph[0, :, :].cpu())) * self.losslr2

        loss2 = masked_softmax_cross_entropy(self.outputs, self.labels,
                                             self.labels_mask)

        loss = loss1 + loss2
        # test_feat = self.x.cpu().detach().numpy()
        # test_D = D.cpu().detach().numpy()
        # test_S = self.sgraph.cpu().detach().numpy()
        return loss, loss1, loss2

    def val(self):

        # D = torch.matrix_diag(torch.ones(self.num_nodes)) * -1
        D = torch.diag(torch.ones(self.num_nodes)) * -1
        D = (D + self.sgraph[0, :, :].cpu()) * -1
        # D = torch.matmul(torch.transpose(self.x[0, :, :]), D)
        # D = torch.matmul(self.x[0, :, :].T.cpu(), D)
        D = torch.matmul(self.inputs[0, :, :].T.cpu(), D)
        # loss1 = torch.trace(torch.matmul(D, self.x[0, :, :].cpu())) * self.losslr1
        loss1 = torch.trace(torch.matmul(D, self.inputs[0, :, :].cpu())) * self.losslr1
        loss1 -= torch.trace(torch.matmul(self.sgraph[0, :, :].cpu().T, self.sgraph[0, :, :].cpu())) * self.losslr2

        loss2 = masked_softmax_cross_entropy(self.outputs, self.val_labels,
                                             self.val_mask)


        loss = loss1 + loss2

        acc = masked_accuracy(self.outputs, self.val_labels,
                              self.val_mask)

        return [loss2, acc]

    def test(self):
        acc = masked_accuracy(self.outputs, self.test_labels,
                              self.test_mask)

        return acc



class GcnEncoder(nn.Module):

    """
    https://github.com/RexYing/gnn-model-explainer
    """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims=[],
            concat=True,
            bn=True,
            dropout=0.0,
            args=None,
    ):
        super(GcnEncoder, self).__init__()

        self.bn = bn
        self.bias = True
        self.gpu = args.gpu
        self.num_aggs = 1
        self.act = nn.ReLU()

        # loss
        if hasattr(args, "loss_weight"):
            print("Loss weight: ", args.loss_weight)
            self.celoss = nn.CrossEntropyLoss(weight=args.loss_weight)
        else:
            self.celoss = nn.CrossEntropyLoss()

        # layers
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            normalize=True,
            dropout=dropout
        )

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs
        )


        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            normalize=False,
            dropout=0.0,
    ):
        conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
        )
        conv_block = nn.ModuleList(
            [
                GraphConv(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    normalize_embedding=normalize,
                    dropout=dropout,
                    bias=self.bias,
                    gpu=self.gpu,
                )
                for i in range(num_layers - 2)
            ]
        )
        conv_last = GraphConv(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
        )
        return conv_first, conv_block, conv_last

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.adj_atts = []
        self.embedding_tensor, adj_att = self.gcn_forward(
            x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
        )
        pred = self.pred_model(self.embedding_tensor)
        return pred, adj_att


    def gcn_forward(
        self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None
    ):

        """ Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
            The embedding dim is self.pred_input_dim
        """

        x, adj_att = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        adj_att_all = [adj_att]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x, _ = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(adj_att)
        x, adj_att = conv_last(x, adj)
        x_all.append(x)
        adj_att_all.append(adj_att)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        return x_tensor, adj_att_tensor

    def construct_mask(self, max_nodes, batch_num_nodes):
        """ For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        """
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def build_pred_layers(
        self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1
    ):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            if not self.gpu:
                pred_model = nn.Linear(pred_input_dim, label_dim)
            else:
                pred_model = nn.Linear(pred_input_dim, label_dim).cuda()

        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:

                if not self.gpu:
                    pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                else:
                    pred_layers.append(nn.Linear(pred_input_dim, pred_dim).cuda())
                pred_layers.append(self.act)
                pred_input_dim = pred_dim

            if not self.gpu:
                pred_layers.append(nn.Linear(pred_dim, label_dim))
            else:
                pred_layers.append(nn.Linear(pred_dim, label_dim).cuda())

            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def loss(self, pred, label):
        pred = torch.transpose(pred, 1, 2)
        return self.celoss(pred, label)



class GlcnEncoder(nn.Module):
    """
    modify from: https://github.com/jiangboahu/GLCN-tf/blob/master/glcn/models.py
    """
    def __init__(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims=[],
            concat=True,
            bn=True,
            dropout=0.0,
            args=None,
            edge = None
    ):
        super(GlcnEncoder, self).__init__()

        self.dropout = args.dropout
        self.bn = args.bn
        self.bias = True
        self.gpu = args.gpu
        self.num_aggs = 1
        self.act = nn.ReLU()

        self.args = args

        self.losslr1 = args.losslr1
        self.losslr2 = args.losslr2

        self.edge = edge
        # loss
        if hasattr(args, "loss_weight"):
            print("Loss weight: ", args.loss_weight)
            self.celoss = nn.CrossEntropyLoss(weight=args.loss_weight)
        else:
            self.celoss = nn.CrossEntropyLoss()

        # layers
        self.conv_gl, self.conv_first, self.conv_block, self.conv_last = self.build_gl_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            normalize=True,
            dropout=dropout
        )

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs
        )

        for m in self.modules():
            if isinstance(m, SparseGraphLearn):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

                if m.a is not None:
                    init.xavier_uniform_(m.a.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))

            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
                    # init.xavier_uniform_(m.bias.data.unsqueeze(0), gain=nn.init.calculate_gain("relu"))


    def build_gl_conv_layers(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            normalize=False,
            dropout=0.0,
    ):

        conv_gl = SparseGraphLearn(
            input_dim=input_dim,
            output_dim=hidden_dim,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            act=torch.nn.ReLU(),
            dropout=self.dropout,
            edge = self.edge)

        conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
        )

        conv_block = nn.ModuleList(
            [
                GraphConv(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    normalize_embedding=normalize,
                    dropout=dropout,
                    bias=self.bias,
                    gpu=self.gpu,
                )
                for i in range(num_layers - 2)
            ]
        )
        conv_last = GraphConv(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
        )
        return conv_gl, conv_first, conv_block, conv_last

    def forward(self, x, adj, batch_num_nodes=None, type = "train",**kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.adj_atts = []
        if type == "train":
            self.embedding_tensor, adj_att = self.glcn_forward(
                x, adj, self.conv_gl, self.conv_first, self.conv_block, self.conv_last, embedding_mask
            )
        else:
            self.embedding_tensor, adj_att = self.gcn_forward(
                x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
            )
        pred = self.pred_model(self.embedding_tensor)
        return pred, adj_att

    def glcn_forward(
            self, x, adj, conv_gl, conv_first, conv_block, conv_last, embedding_mask=None
    ):

        """ Perform forward prop with graph learn convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
            The embedding dim is self.pred_input_dim
        """
        self.x = x
        # x, adj_att = conv_first(x, adj)
        # h, sgraph = conv_gl(x, adj)
        h, sgraph = conv_gl(x)
        self.sgraph = sgraph
        self.num_nodes = adj.shape[1]

        x, _ = conv_first(x, sgraph)

        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # x_all = []
        adj_att_all = [sgraph]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            # x, _ = conv_block[i](x, adj)
            x, _ = conv_block[i](x, sgraph)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(sgraph)
        x, adj_att = conv_last(x, sgraph)
        x_all.append(x)
        adj_att_all.append(adj_att)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        return x_tensor, sgraph

    def gcn_forward(
                self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None
        ):

            """ Perform forward prop with graph learn convolution.
            Returns:
                Embedding matrix with dimension [batch_size x num_nodes x embedding]
                The embedding dim is self.pred_input_dim
            """

            x, adj_att = conv_first(x, adj)

            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all = [x]
            # x_all = []
            adj_att_all = [adj_att]
            # out_all = []
            # out, _ = torch.max(x, dim=1)
            # out_all.append(out)
            for i in range(len(conv_block)):
                # x, _ = conv_block[i](x, adj)
                x, _ = conv_block[i](x, adj)
                x = self.act(x)
                if self.bn:
                    x = self.apply_bn(x)
                x_all.append(x)
                adj_att_all.append(adj)
            x, adj_att = conv_last(x, adj)
            x_all.append(x)
            adj_att_all.append(adj_att)
            # x_tensor: [batch_size x num_nodes x embedding]
            x_tensor = torch.cat(x_all, dim=2)
            if embedding_mask is not None:
                x_tensor = x_tensor * embedding_mask
            self.embedding_tensor = x_tensor

            # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
            adj_att_tensor = torch.stack(adj_att_all, dim=3)
            return x_tensor, adj_att_tensor

    def construct_mask(self, max_nodes, batch_num_nodes):
        """ For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        """
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def build_pred_layers(
            self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1
    ):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            if not self.gpu:
                pred_model = nn.Linear(pred_input_dim, label_dim)
            else:
                pred_model = nn.Linear(pred_input_dim, label_dim).cuda()

        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:

                if not self.gpu:
                    pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                else:
                    pred_layers.append(nn.Linear(pred_input_dim, pred_dim).cuda())
                pred_layers.append(self.act)
                pred_input_dim = pred_dim

            if not self.gpu:
                pred_layers.append(nn.Linear(pred_dim, label_dim))
            else:
                pred_layers.append(nn.Linear(pred_dim, label_dim).cuda())

            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def loss(self, pred, label):
        pred = torch.transpose(pred, 1, 2)
        sl = self.sparse_loss()
        return sl + self.celoss(pred, label)

    def sparse_loss(self):

        # D = torch.matrix_diag(torch.ones(self.num_nodes)) * -1
        D = torch.diag(torch.ones(self.num_nodes)) * -1
        D = (D + self.sgraph[0, :, :].cpu()) * -1
        # D = torch.matmul(torch.transpose(self.x[0, :, :]), D)
        D = torch.matmul(self.x[0, :, :].T.cpu(), D)
        loss1 = torch.trace(torch.matmul(D, self.x[0, :, :].cpu())) * self.losslr1
        loss1 -= torch.trace(torch.matmul(self.sgraph[0, :, :].cpu().T,
                                                                       self.sgraph[0, :, :].cpu()))* self.losslr2
        return loss1



use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class Explainer:
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        train_idx,
        args,
        writer=None,
        print_training=True,
        graph_mode=False,
        graph_idx=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training


    # Main method
    def explain(
        self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp", path_mode=False
    ):
        """Explain a single node prediction
        """

        if path_mode:
            mask_list = []

        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            neighbors = np.asarray(range(self.adj.shape[0]))
        elif path_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat
            sub_label = self.label
            neighbors = np.asarray(range(self.adj.shape[0]))

        else:
            print("node label: ", self.label[graph_idx][node_idx])
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            print("neigh graph idx: ", node_idx, node_idx_new)
            sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)

        if not path_mode:
            sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            print("Graph predicted label: ", pred_label)
        elif path_mode:
            pred_label = np.argmax(self.pred.detach().cpu(), axis=1)

        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Node predicted label: ", pred_label[node_idx_new])

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()


        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )
                single_subgraph_label = sub_label.squeeze()
                #
                # if self.writer is not None:
                #     self.writer.add_scalar("mask/density", mask_density, epoch)
                #     self.writer.add_scalar(
                #         "optimization/lr",
                #         explainer.optimizer.param_groups[0]["lr"],
                #         epoch,
                #     )
                if epoch % 25 == 0:
                    feat_mask, mask_adj = explainer.log_mask(epoch)
                    explainer.log_masked_adj(
                        node_idx_new, epoch, label=single_subgraph_label
                    )
                    if path_mode:
                        mask_list.append([epoch, feat_mask, mask_adj])

                        # explainer.log_adj_grad(
                        #     node_idx_new, pred_label, epoch, label=single_subgraph_label
                        # )

                    # if epoch == 0:
                        # if self.model.att:
                        #     # explain node
                        #     print("adj att size: ", adj_atts.size())
                        #     adj_att = torch.sum(adj_atts[0], dim=2)
                        #     # adj_att = adj_att[neighbors][:, neighbors]
                        #     node_adj_att = adj_att * adj.float().cuda()
                        #     io_utils.log_matrix(
                        #         self.writer, node_adj_att[0], "att/matrix", epoch
                        #     )
                        #     node_adj_att = node_adj_att[0].cpu().detach().numpy()
                        #     G = io_utils.denoise_graph(
                        #         node_adj_att,
                        #         node_idx_new,
                        #         threshold=3.8,  # threshold_num=20,
                        #         max_component=True,
                        #     )
                        #     io_utils.log_graph(
                        #         self.writer,
                        #         G,
                        #         name="att/graph",
                        #         identify_self=not self.graph_mode,
                        #         nodecolor="label",
                        #         edge_vmax=None,
                        #         args=self.args,
                        #     )
                if model != "exp":
                    break

            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()

        # fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
        #         'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')
        # with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
        #     np.save(outfile, np.asarray(masked_adj.copy()))
        #     print("Saved adjacency matrix to ", fname)
        if path_mode:
            masked_adj = mask_list

        return masked_adj

    # path explain
    def explain_path_stats(self, node_indices, args, graph_idx=0, model="exp"):
        masked_adjs = self.explain(node_indices, graph_idx=graph_idx, model=model, path_mode=True)
        #
        # if not args.method == "pglcn":
        #     # pdb.set_trace()
        #     graphs = []
        #     feats = []
        #     adjs = []
        #     pred_all = []
        #     real_all = []
        #     for i, idx in enumerate(node_indices):
        #         new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
        #         G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
        #         pred, real = self.make_pred_real(masked_adjs[i], new_idx)
        #         pred_all.append(pred)
        #         real_all.append(real)
        #         denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
        #         # denoised_adj = nx.to_numpy_matrix(G)
        #         denoised_adj = nx.to_numpy_array(G)
        #         graphs.append(G)
        #         feats.append(denoised_feat)
        #         adjs.append(denoised_adj)
        #         io_utils.log_graph(
        #             self.writer,
        #             G,
        #             "graph/{}_{}_{}".format(self.args.dataset, model, i),
        #             identify_self=True,
        #             args=self.args
        #         )
        #
        #     pred_all = np.concatenate((pred_all), axis=0)
        #     real_all = np.concatenate((real_all), axis=0)
        #
        #     auc_all = roc_auc_score(real_all, pred_all)
        #     precision, recall, thresholds = precision_recall_curve(real_all, pred_all)
        #
        #     plt.switch_backend("agg")
        #     plt.plot(recall, precision)
        #     # plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")
        #     plt.savefig("log/pr_" + self.args.dataset + "_" + model + ".png")
        #
        #     plt.close()

            # auc_all = roc_auc_score(real_all, pred_all)
            # precision, recall, thresholds = precision_recall_curve(real_all, pred_all)
            #
            # plt.switch_backend("agg")
            # plt.plot(recall, precision)
            # # plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")
            # plt.savefig("log/pr_" + self.args.dataset + "_" + model + ".png")
            #
            # plt.close()

            # with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            # with open("log/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            #     f.write(
            #         "dataset: {}, model: {}, auc: {}\n".format(
            #             self.args.dataset, "exp", str(auc_all)
            #         )
            #     )

        return masked_adjs


    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes

        Args:
            - node_indices  :  Indices of the nodes to be explained
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs


    def explain_nodes_gnn_stats(self, node_indices, args, graph_idx=0, model="exp"):
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        for i, idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
            pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            pred_all.append(pred)
            real_all.append(real)
            denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            # denoised_adj = nx.to_numpy_matrix(G)
            denoised_adj = nx.to_numpy_array(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            # io_utils.log_graph(
            #     self.writer,
            #     G,
            #     "graph/{}_{}_{}".format(self.args.dataset, model, i),
            #     identify_self=True,
            #     args=self.args
            # )

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.close()
        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/"+ self.args.dataset + "_" + "explain/" + "pr_" + self.args.method + ".png")
        plt.close()
        # plt.savefig("log/pr_" + self.args.dataset + "_" + model + ".png")

        # plt.close()

        # auc_all = roc_auc_score(real_all, pred_all)
        # precision, recall, thresholds = precision_recall_curve(real_all, pred_all)
        #
        # plt.switch_backend("agg")
        # plt.plot(recall, precision)
        # # plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")
        # plt.savefig("log/pr_" + self.args.dataset + "_" + model + ".png")
        #
        # plt.close()

        # with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
        with open("log/" + self.args.dataset + "_" + "explain/" + args.method + "_auc_"  + model + ".txt", "w") as f:
            f.write(
                "dataset: {}, model: {}, auc: {}\n".format(
                    self.args.dataset, "exp", str(auc_all)
                )
            )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices):
        """
        Explain graphs.
        """
        masked_adjs = []

        for graph_idx in graph_indices:
            masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
            G_denoised = io_utils.denoise_graph(
                masked_adj,
                0,
                threshold_num=20,
                feat=self.feat[graph_idx],
                max_component=False,
            )
            label = self.label[graph_idx]
            io_utils.log_graph(
                self.writer,
                G_denoised,
                "graph/graphidx_{}_label={}".format(graph_idx, label),
                identify_self=False,
                nodecolor="feat",
                args=self.args
            )
            masked_adjs.append(masked_adj)

            G_orig = io_utils.denoise_graph(
                self.adj[graph_idx],
                0,
                feat=self.feat[graph_idx],
                threshold=None,
                max_component=False,
            )

            io_utils.log_graph(
                self.writer,
                G_orig,
                "graph/graphidx_{}".format(graph_idx),
                identify_self=False,
                nodecolor="feat",
                args=self.args
            )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")

        return masked_adjs

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        print(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    idx, graph_idx
                )
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                # node_color='#336699',

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh", tensorboardX.utils.figure_to_image(fig), 0
        )

    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        https://papers.nips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        self.alpha = self.preds_grad


    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
        self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs.
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            print("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == "syn1" or self.args.dataset == "syn2":
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        if self.args.dataset == "syn3" or self.args.dataset == "syn5":
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            if real[start - 1][start] > 0:
                real[start - 1][start] = 10
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 5][start + 6] > 0:
                real[start + 5][start + 6] = 10
            if real[start + 6][start + 7]:
                real[start + 6][start + 7] = 10
            if real[start - 1][start + 2]:
                real[start - 1][start + 2] = 10
            if real[start + 2][start + 5]:
                real[start + 2][start + 5] = 10
            if real[start + 0][start + 3]:
                real[start + 0][start + 3] = 10
            if real[start + 3][start + 6]:
                real[start + 3][start + 6] = 10
            if real[start + 4][start + 7]:
                real[start + 4][start + 7] = 10

            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        # if args.method == "pglcn":
            # self.feat_mask = self.construct_feat_mask((x.size()[1],x.size()[2]), init_strategy="constant")
        # else:
        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")

        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        if args.method == "pglcn":
            self.coeffs = {
                "size": 2e-5,
                "feat_size": 1.0,
                "ent": 1.0,
                "feat_ent": 0.1,
                "grad": 0,
                "lap": 1.0,
            }
        else:
            self.coeffs = {
                "size": 0.005,
                "feat_size": 1.0,
                "ent": 1.0,
                "feat_ent": 0.1,
                "grad": 0,
                "lap": 1.0,
            }
    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        # if self.args.method == "pglcn":
        #     mask = nn.Parameter(torch.FloatTensor(feat_dim[0],feat_dim[1]))
        # else:
        mask = nn.Parameter(torch.FloatTensor(feat_dim))

        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False, mask_features=True, marginalize=False):
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask
        if self.args.method=="pglcn":
            ypred = self.model(feat=x, adj=self.masked_adj, type="explain")
            adj_att = ypred

        elif self.args.method=="glcn":
            ypred, adj_att = self.model(x, self.masked_adj, type="explain")
        else:
            ypred, adj_att = self.model(x, self.masked_adj)
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        elif self.args.method=="pglcn":
            res = nn.Softmax(dim=0)(ypred)
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))

        elif self.args.method=="pglcn":
            if self.args.gpu:
                self.label = self.label.cuda()

            if self.label.size()[1] == 1:
                y = self.label.type(torch.int64)
                y_one_hot = torch.ones(self.label.size()[0], 2).cuda()
                y_one_hot = y_one_hot.scatter_(1, y, 0)
                labels = y_one_hot
            else:
                labels = self.label

            pred_loss = torch.mean( torch.nn.CrossEntropyLoss(reduction="none")(pred, labels))

        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0

        elif self.args.method=="pglcn":
            lap_loss = 0

        else:
            lap_loss = (self.coeffs["lap"]
                * (pred_label_t @ L @ pred_label_t)
                / self.adj.numel()
            )

        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)
        # if self.args.method == "pglcn":
        #     loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss + feat_mask_ent_loss
        # else:
        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss

        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    def log_mask(self, epoch):
        # plt.switch_backend("agg")
        # fig = plt.figure(figsize=(4, 3), dpi=400)
        # plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")
        #
        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image(
        #     "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        # )

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        # io_utils.log_matrix(
        #     self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        # )
        feat_mask = torch.sigmoid(self.feat_mask)
        mask_adj = self.masked_adj[0].cpu().detach().numpy()
        # fig = plt.figure(figsize=(4, 3), dpi=400)
        # # use [0] to remove the batch dim
        # plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")
        #
        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image(
        #     "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        # )
        #
        # if self.args.mask_bias:
        #     fig = plt.figure(figsize=(4, 3), dpi=400)
        #     # use [0] to remove the batch dim
        #     plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        #     cbar = plt.colorbar()
        #     cbar.solids.set_edgecolor("face")
        #
        #     plt.tight_layout()
        #     fig.canvas.draw()
        #     self.writer.add_image(
        #         "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
        #     )
        return feat_mask, mask_adj

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print("GRAPH model")
            G = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

        # if graph attention, also visualize att

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        elif self.args.method=="pglcn":
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=None,
                threshold=0.1,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )

