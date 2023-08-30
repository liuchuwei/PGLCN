import torch.nn
from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Ridge, ElasticNet, Lasso, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


from torch.nn import init
from model.layers import *
from model.metrics import *

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
            dropout=0.0,
        )

        self.conv_last = GraphConvSlice(
            input_dim=hidden_gcn,
            output_dim=self.output_dim,
            normalize_embedding=normalize,
            bias=False,
            gpu=self.gpu,
            act=lambda x: x,
            dropout=0.0,
        )

        return self.conv_gl, self.conv_first, self.conv_last

    def build_pred_layers(self):

        pred_layers = []
        drop_rate = self.args.dropout2
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