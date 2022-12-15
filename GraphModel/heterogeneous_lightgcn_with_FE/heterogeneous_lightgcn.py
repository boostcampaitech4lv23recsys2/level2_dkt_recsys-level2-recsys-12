from typing import Optional, Union
"""
Union:
    여러 개의 타입이 허용될 수 있는 상황에서는 typing 모듈의 
    Union을 사용할 수 있습니다.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor

from config import CFG

class Heterogeneous_LightGCN(torch.nn.Module):
    def __init__(
        self,
        # user_problem_nodes: 노드로 userID와 assessmentItemID을 갖는 bipartite graph
            # 다시 이야기하면 user와 problem을 고려한 message passing을 수행한다는 의미
        user_problem_nodes: int,
        
        # user_test_nodes: 노드로 user와 testId를 갖는 bipartite graph
            # 다시 이야기하면 user와 test를 고려한 message passing을 수행한다는 의미
        user_test_nodes: int,
        
        # user_tag_nodes: 노드로 user와 KnowledgeTag를 갖는 bipartite graph
            # 다시 이야기하면 user와 tag를 고려한 message passing을 수행한다는 의미
        user_tag_nodes: int,
        
        embedding_dim: int,
        num_user_feature: int,
        num_item_feature: int,
        feature_aggregation_method: int = 0,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.user_problem_nodes = user_problem_nodes
        self.user_tag_nodes = user_tag_nodes
        self.user_test_nodes = user_test_nodes
        self.embedding_dim = embedding_dim
        
        self.feature_aggregation_method = feature_aggregation_method
        
        # 아래의 LGConv: 이 부분은 파라미터 없이 작동하고, 별도로 설정할 것이 없기에 웬만하면 건들지 않는게 좋습니다.
        self.user_problem_embedding = Embedding(self.user_problem_nodes, embedding_dim)
        # self.user_problem_convs: user-problem의 bipartite 그래프 message passing
        self.user_problem_convs = ModuleList([LGConv(**kwargs) for _ in range(CFG.feature_num_layers[0])])
        
        self.user_test_embedding = Embedding(self.user_test_nodes, embedding_dim)
        # self.user_test_convs: user-test의 bipartite 그래프 message passing
        self.user_test_convs = ModuleList([LGConv(**kwargs) for _ in range(CFG.feature_num_layers[1])])
        
        self.user_tag_embedding = Embedding(self.user_tag_nodes, embedding_dim)
        # self.user_tag_convs: user-tag의 bipartite 그래프 message passing
        self.user_tag_convs = ModuleList([LGConv(**kwargs) for _ in range(CFG.feature_num_layers[2])])
        
        # weight의 초기값을 별도로 구성했습니다.
        self.reset_parameters()
        
        # forward 함수에서 feature_aggregation_method == 2: # nn.Linear_method 사용할 때 사용함
        self.out_src_linear = torch.nn.Linear(3 * embedding_dim, embedding_dim)
        self.out_dst_linear = torch.nn.Linear(3 * embedding_dim, embedding_dim)
        
        # feature별로 feature engineering들을 임베딩 후 합치기 == nn.Linear(feature 수, emb_dim)
        self.num_user_feature = num_user_feature
        self.num_item_feature = num_item_feature
        self.out_src_feature_engineering_linear = torch.nn.Linear(self.num_user_feature, embedding_dim)
        self.out_dst_feature_engineering_linear = torch.nn.Linear(self.num_item_feature, embedding_dim)
        
        # forward 함수에서 feature_aggregation_method == 3: # attention_method 사용할 때 사용함
        self.src_query = torch.nn.Linear(self.embedding_dim, embedding_dim)
        self.src_key = torch.nn.Linear(self.embedding_dim, embedding_dim)
        self.src_value = torch.nn.Linear(self.embedding_dim, embedding_dim)
        self.src_attn = torch.nn.MultiheadAttention(embedding_dim, 1)
        
        self.dst_query = torch.nn.Linear(self.embedding_dim, embedding_dim)
        self.dst_key = torch.nn.Linear(self.embedding_dim, embedding_dim)
        self.dst_value = torch.nn.Linear(self.embedding_dim, embedding_dim)
        self.dst_attn = torch.nn.MultiheadAttention(embedding_dim, 1)
        
    # predict_link()하는 부분에서 sigmoid를 사용하기에, xavier_uniform_으로 초기화하는 것이 좋습니다.
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.user_problem_embedding.weight)
        torch.nn.init.xavier_uniform_(self.user_test_embedding.weight)
        torch.nn.init.xavier_uniform_(self.user_tag_embedding.weight)
        for conv in self.user_problem_convs:
            conv.reset_parameters()
        for conv in self.user_test_convs:
            conv.reset_parameters()
        for conv in self.user_tag_convs:
            conv.reset_parameters()
            
    """
        edge_index:
            from torch_geometric.typing import Adj, OptTensor 를 보니, custom 타입임을 알 수 있습니다.
            실제 사용은, forward에서 train_data["edge"]를 사용합니다.
            train_data_proc: {"edge":[유저,문제], "label":문제 맞췄는지} 형태 참고
            n by 2 행렬
    """
    def get_embedding(
        self,
        user_problem_edge_index: Adj,
        user_test_edge_index: Adj,
        user_tag_edge_index: Adj,
        ) -> Tensor:
        
        user_problem_x = self.user_problem_embedding.weight
        user_test_x = self.user_test_embedding.weight
        user_tag_x = self.user_tag_embedding.weight
        
        user_problem_out = user_problem_x * (1 / (CFG.feature_num_layers[0] + 1))
        user_test_out = user_test_x * (1 / (CFG.feature_num_layers[1] + 1))
        user_tag_out = user_tag_x * (1 / (CFG.feature_num_layers[2] + 1))
        
######## message passing 단계입니다. ########
        # user와 problem에 대한 bipartite 그래프 message passing입니다.
        for i in range(CFG.feature_num_layers[0]):
            # edge_index 사이즈가 2 by N 이어야함
            user_problem_x = self.user_problem_convs[i](user_problem_x, user_problem_edge_index)
            user_problem_out = user_problem_out + user_problem_x * (1 / (CFG.feature_num_layers[0] + 1))
            
        # user와 test에 대한 bipartite 그래프 message passing입니다.
        for i in range(CFG.feature_num_layers[1]):
            # edge_index 사이즈가 2 by N 이어야함
            user_test_x = self.user_test_convs[i](user_test_x, user_test_edge_index)
            user_test_out = user_test_out + user_test_x * (1 / (CFG.feature_num_layers[1] + 1))
            
        # user와 tag에 대한 bipartite 그래프 message passing입니다.
        for i in range(CFG.feature_num_layers[2]):
            # edge_index 사이즈가 2 by N 이어야함
            user_tag_x = self.user_tag_convs[i](user_tag_x, user_tag_edge_index)
            user_tag_out = user_tag_out + user_tag_x * (1 / (CFG.feature_num_layers[2] + 1))
            
        return user_problem_out, user_test_out, user_tag_out

    def forward(
            self,
            user_problem_edge_index: Adj,
            user_test_edge_index: Adj,
            user_tag_edge_index: Adj,
            out_src_feature_engineering: Tensor,
            out_dst_feature_engineering: Tensor,) -> Tensor:
        
        # torch.Size([문제풀이 기록 수, n_feature]) -> torch.Size([문제풀이 기록 수, emb_dim])
        out_src_feature_engineering = self.out_src_feature_engineering_linear(out_src_feature_engineering)
        out_dst_feature_engineering = self.out_dst_feature_engineering_linear(out_dst_feature_engineering)
        
        # user_feature_out.shape:  torch.Size([userId 개수 + feature의 category 개수, embedding_dim])
        user_problem_out, user_test_out, user_tag_out = self.get_embedding(user_problem_edge_index,
                                            user_test_edge_index,
                                            user_tag_edge_index)

        # 참고: problem을 고려한 user embedding, test를 고려한 user embedding, tag를 고려한 user embedding 은 모두 별개의 정보이다.
        # out_src: problem을 고려한 user embedding, test를 고려한 user embedding, tag를 고려한 user embedding의 aggregation
        # out_dst: user를 고려한 problem embedding, user를 고려한 test embedding, user를 고려한 tag embedding의 aggregation
        
        if self.feature_aggregation_method == 0: # sum_method
            """ 
                sum_method란,
                    user_feature_out[user_feature_edge_index[0]].shape:
                        torch.Size([문제풀이 기록 수 == train data 수, embedding_dim])
                        -> sum으로 이것들을 aggregation 하는 방법입니다.
            """
            out_src = (1/3) * (user_problem_out[user_problem_edge_index[0]]
                    + user_test_out[user_test_edge_index[0]]
                    + user_tag_out[user_tag_edge_index[0]])
            out_src += out_src_feature_engineering
            
            out_dst = (1/3) * (user_problem_out[user_problem_edge_index[1]]
                + user_test_out[user_test_edge_index[1]]
                + user_tag_out[user_tag_edge_index[1]])
            out_dst += out_dst_feature_engineering
            
        elif self.feature_aggregation_method == 1: # element_wise_method
            """ 
                element_wise_method란,
                    user_feature_out[user_feature_edge_index[0]].shape:
                        torch.Size([문제풀이 기록 수 == train data 수, embedding_dim])
                        -> element_wise로 이것들을 aggregation 하는 방법입니다.
            """
            out_src = (user_problem_out[user_problem_edge_index[0]]
                    * user_test_out[user_test_edge_index[0]]
                    * user_tag_out[user_tag_edge_index[0]])
            out_src += out_src_feature_engineering
            out_src = (user_problem_out[user_problem_edge_index[1]]
                    * user_test_out[user_test_edge_index[1]]
                    * user_tag_out[user_tag_edge_index[1]])
            out_dst += out_dst_feature_engineering
            
        elif self.feature_aggregation_method == 2: # nn.Linear_method
            """
                nn.Linear_method란,
                    user_feature_out[user_feature_edge_index[0]].shape:
                        torch.Size([문제풀이 기록 수 == train data 수, embedding_dim])
                        -> nn.Linear로 이것들을 aggregation 하는 방법입니다.
                    nn.Linear_method 진행 과정:
                        1. user_feature_out들 concat:
                            torch.Size([feature 개수, 문제풀이 기록 수 == train data 수, embedding_dim])
                        2. nn.view(문제풀이 기록 수 == train data 수, -1):
                            torch.Size([문제풀이 기록 수 == train data 수, feature 개수 * embedding_dim])
                        3. nn.Linear(feature 개수 * embedding_dim, embedding_dim):
                            torch.Size([문제풀이 기록 수 == train data 수, embedding_dim])
            """
            out_src = torch.cat([
                user_problem_out[user_problem_edge_index[0]],
                user_test_out[user_test_edge_index[0]],
                user_tag_out[user_tag_edge_index[0]]
            ], dim=1)
            out_src = self.out_src_linear(out_src)
            out_src += out_src_feature_engineering
            
            out_dst = torch.cat([
                user_problem_out[user_problem_edge_index[1]],
                user_test_out[user_test_edge_index[1]],
                user_tag_out[user_tag_edge_index[1]]
            ], dim=1)
            out_dst = self.out_dst_linear(out_dst)
            out_dst += out_dst_feature_engineering

        elif self.feature_aggregation_method == 3: # attention_method
            out_src = torch.stack([
                user_problem_out[user_problem_edge_index[0]],
                user_test_out[user_test_edge_index[0]],
                user_tag_out[user_tag_edge_index[0]]
            ], dim=0)
            out_src += out_src_feature_engineering
            
            out_dst = torch.stack([
                user_problem_out[user_problem_edge_index[1]],
                user_test_out[user_test_edge_index[1]],
                user_tag_out[user_tag_edge_index[1]]
            ], dim=0)
            out_dst += out_dst_feature_engineering
            
            
            out_src_query = self.src_query(out_src)
            out_src_key = self.src_key(out_src)
            out_src_value = self.src_value(out_src)
            out_src_attn = self.src_attn(out_src_query, out_src_key, out_src_value, need_weights=False)
            out_src = out_src_attn[0].sum(dim=0)
            
            out_dst_query = self.dst_query(out_dst)
            out_dst_key = self.dst_key(out_dst)
            out_dst_value = self.dst_value(out_dst)
            out_dst_attn = self.dst_attn(out_dst_query, out_dst_key, out_dst_value, need_weights=False)
            out_dst = out_dst_attn[0].sum(dim=0)

        # aggregation 메소드에 상관없이 모두 
        # torch.Size([문제풀이 기록 수 == train data 수, embedding_dim]) 크기가 됨
        return (out_src * out_dst).sum(dim=-1)

    def predict_link(
            self,
            user_problem_edge_index: Adj,
            user_test_edge_index: Adj,
            user_tag_edge_index: Adj,
            user_feature_engineering: Tensor,
            item_feature_engineering: Tensor,
            # edge_label_index: OptTensor = None,
            prob: bool = False) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.
        Args:
            prob (bool): Whether probabilities should be returned. (default:
                :obj:`False`)
        """
        pred = self(user_problem_edge_index,
                    user_test_edge_index,
                    user_tag_edge_index,
                    user_feature_engineering,
                    item_feature_engineering).sigmoid()
        # 예측값을 반올림할 것인지 말것인지
        # classification할 것인지, regression할 것인지
        return pred if prob else pred.round()

    # 이진 cross-entropy loss 사용
    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.
        Args:
            pred (Tensor): The predictions.
            edge_label (Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))