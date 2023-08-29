import torch
import torch.nn as nn
import torch.nn.functional as F


# SGCN basic operation
class SparseGraphLearn(nn.Module):
    """
    Sparse Graph learning layer.
    modify from: https://github.com/jiangboahu/GLCN-tf/blob/master/glcn/models.py
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 normalize_embedding=False,
                 dropout=0.0,
                 bias=True,
                 gpu=True,
                 act=torch.nn.ReLU(),
                 edge=None
                 ):
        super(SparseGraphLearn, self).__init__()
        self.act = act
        self.gpu = gpu
        self.dropout = dropout
        self.edge = edge
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding

        if not gpu:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())

        if bias:
            if not gpu:
                self.a = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.a = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.a = None

    def forward(self, x):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        h = torch.matmul(x, self.weight)

        # edge = adj[0, :, :]
        # edge = sp.coo_matrix(edge.detach().cpu())
        #
        # edge0 = edge.col
        # edge1 = edge.row
        #
        # edge_v = torch.abs(torch.gather(h, edge0) - torch.gather(h,edge1))
        edge_v = torch.abs(h[0, self.edge[0], :] - h[0, self.edge[1], :] )
        edge_v = torch.squeeze(self.act(torch.matmul(edge_v, self.a)))
        sgraph = torch.sparse_coo_tensor(self.edge, edge_v, size=[h.shape[1], h.shape[1]])
        sgraph = torch.sparse.softmax(sgraph, 0)
        sgraph = sgraph.to_dense()
        sgraph = sgraph.unsqueeze(0)
        # test = sgraph[0, :, :].cpu().detach().numpy()
        return h, sgraph



# GCN basic operation
class GraphConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        normalize_embedding=False,
        dropout=0.0,
        bias=True,
        gpu=True,
        act=None
    ):
        super(GraphConv, self).__init__()
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.act = act

        if not gpu:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())

        if bias:
            if not gpu:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None


    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)

        if self.bias is not None:
            y = y + self.bias

        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)

        if not self.act is None:
            y = self.act(y)

        return y, adj


class GraphConvSlice(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        normalize_embedding=False,
        dropout=0.0,
        bias=True,
        gpu=True,
        act=None
    ):
        super(GraphConvSlice, self).__init__()
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.act = act

        if not gpu:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())

        if bias:
            if not gpu:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None


    def forward(self, x, adj):

        second = x.clone()
        second = torch.split(second, 1, dim=0)
        def fn(x_slice):

            if self.dropout > 0.001:
                x_slice = self.dropout_layer(x_slice)

            y = torch.matmul(adj, x_slice)
            y = torch.matmul(y, self.weight)

            if self.bias is not None:
                y = y + self.bias

            return y

        output = [fn(item) for item in second]
        out = torch.squeeze(torch.stack(output, dim=1))

        if self.normalize_embedding:
            out = F.normalize(out, p=2, dim=2)

        if not self.act is None:
            out = self.act(out)
        # out = torch.stack(x)
        # out = torch.matmul(out, self.weight)
        return out
