""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
        
        
def compute_correlation_matrix(A, B):
    if A.shape != B.shape:
        raise ValueError("Both matrices should have the same shape")

    # Compute column-wise means and standard deviations
    mean_A = torch.mean(A, dim=0)
    mean_B = torch.mean(B, dim=0)

    std_A = torch.std(A, dim=0, unbiased=False)
    std_B = torch.std(B, dim=0, unbiased=False)

    # Compute the correlation matrix
    A_centered = A - mean_A
    B_centered = B - mean_B

    numerator = torch.mm(A_centered.T, B_centered) / A.shape[0]
    denominator = std_A[:, None] * std_B[None, :]

    correlation_matrix = numerator / denominator

    return correlation_matrix
           
    
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        
        return x


class Attention(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.query = nn.Linear(hdim, hdim)
        self.key = nn.Linear(hdim, hdim)
        self.value = nn.Linear(hdim, hdim)

    def forward(self, Q, K, V):
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        attention_weights = F.softmax(Q.mm(K.t()) / (K.size(-1) ** 0.5), dim=-1)
        output = attention_weights.mm(V)
        
        return output

    
class CrossAttention(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.attention_layer = Attention(hdim)

    def forward(self, target_matrix, all_matrices):

        attention_outputs = []
        
        # Iterate over all matrices in the list
        for matrix in all_matrices:
            # Skip the target matrix
            if torch.equal(matrix, target_matrix):
                continue

            output = self.attention_layer(matrix, target_matrix, target_matrix) # cross attention
            attention_outputs.append(output)
        
        attention_outputs.append(target_matrix) # add target matrix together
        
#         combined_output = torch.mean(torch.stack(attention_outputs), dim=0) # average
#         combined_output = sum(attention_outputs) # sum
        combined_output = torch.cat(attention_outputs, dim=1) # catenate
        
        return combined_output


class CA_E(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.ca = CrossAttention(hdim)

    def forward(self, target, x_all):
        x = self.ca(target, x_all)
        
        return x
    
    
class SA_E(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.sa = Attention(hdim)

    def forward(self, x):
        x = self.sa(x, x, x)
        
        return x

    
class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hidden_dim, hvcdn_dim): # hidden_dim 
        super().__init__()
        self.num_cls = num_cls
        self.CA_E = CA_E(hidden_dim)
#         self.Classifier_1 = Classifier_1(hidden_dim, num_cls)
#         self.model = nn.Sequential(
#             nn.Linear(pow(num_cls, num_view), hvcdn_dim),
#             nn.LeakyReLU(0.25),
#             nn.Linear(hvcdn_dim, num_cls)
#         )
        self.Classifier_1 = Classifier_1(num_view*hidden_dim, 10*num_cls)
        self.model = nn.Sequential(
            nn.Linear(10*num_cls, hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        
        weight_in_list = []
        for i in range(num_view):
            weight_matrix = self.CA_E(in_list[i], in_list)
            weight_matrix = self.Classifier_1(weight_matrix)
            weight_in_list.append(weight_matrix)
        in_list = weight_in_list
            
#         weight_in_list = []
#         for i, matrix in enumerate(in_list):
#             other_matrices = [t for j, t in enumerate(in_list) if j != i] # (e.g., matrix 1 & 2, matrix 1 & 3) 
#             # Compute the sum of the correlation matrices with the others (e.g., sum = corrmat (1 & 2) + corrmat (1 & 3)) 
#             weight = sum(compute_correlation_matrix(matrix, other) for other in other_matrices)
#             # Multiply matrix with the sum of its correlation matrices with the others (e.g., matrix 1 * sum)
#             weight_matrix =  torch.matmul(matrix, weight)
#             weight_in_list.append(weight_matrix)
#         in_list = weight_in_list
       
#         for i in range(num_view):
#             in_list[i] = torch.sigmoid(in_list[i])
#         x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
#         for i in range(2,len(in_list)):
#             x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
#         vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
#         output = self.model(vcdn_feat)

        output = self.model(sum(in_list))

        return output

    
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, dropout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, dropout)
#         model_dict["S{:}".format(i+1)] = SA_E(dim_he_list[-1])
#         model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_he_list[-1], dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters()),
#                +list(model_dict["S{:}".format(i+1)].parameters()),
#                +list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict
