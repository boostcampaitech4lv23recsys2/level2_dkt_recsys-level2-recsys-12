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


class LightGCN(torch.nn.Module):
    """
        The LightGCN model from the `"LightGCN: Simplifying and Powering
        Graph Convolution Network for Recommendation"
        from <https://arxiv.org/abs/2002.02126> paper.
        num_nodes:
            유저-문제 그래프인 경우: (모든 유저 + 모든 문제) 개수 == 해당 bipartite 그래프에서 가능한 노드들의 총 개수
            
        embedding_dim:
            Embedding layer별로 user_embedding, item_embedding을 생성하는데,
            이때 각 embedding vector의 크기를 어떻게 지정할 것인가?
        num_layers:
            여기서 layer라고 하면, k-hop을 말한다.
            즉, 주변 노드 정보를 얼마나 더 볼 것인가를 의미한다.
            논문 실험 결과에 의하면, layer를 적당히 (1~4) 사이로 쌓는 것이 효과적이라고 함.
            num_layers 파라미터가 과하면 쌓으면 over-smoothing할 수 있음.
        alpha:
            LightGCN은 각 layer들을 통과해서 생긴 embedding vector들을 수합하기 위해,
            weighted sum을 한다.alpha * (l_1 + l_2 + ... + l_k)
            논문에서 weighted sum이라고 부르는데, 학습되는 파라미터는 아니고 hyper-parameter이다.
            보통은 1 / (num_layers + 1)로 한다. (alpha = None으로 설정하면 됨)
            Union[float, Tensor]에 대한 설명은 아래 65번째 줄에 작성했습니다.
            [1,2,3,4]
        **kwargs:
            LightGCN 베이스라인에서, **CFG.build_kwargs 부분이 있는데,
            이는 from torch_geometric.nn.conv import LGConv 에서 LGConv의 parameter를 말한다.
            특별한 파라미터도 없어서 별도로 구성할 필요가 없음.
            그냥 비워두면 내부에서 잘 작동함.
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # layer combination(weighted sum) 단계에서의 weight가 alpha임
        if alpha is None:
            alpha = 1. / (num_layers + 1)

        """
            alpha를 Tensor로 구성했다는 뜻은, 각 layer별로 별도로 alpha 값을 지정하려는 것과 같습니다.
            이때, Tensor크기가 (layer개수 + 1)을 만족하지 않으면 오류가 발생합니다.
            (+ 1을 하는 이유는 layer 통과 전의 embedding vector를 고려하기 위함입니다.)
                alpha 구성 자체를 하이퍼파라미터로 지정할 수 있다는 뜻입니다.
                참고: 논문에서는 모든 layer에 alpha를 1. / (num_layers + 1)로 지정했습니다.
        """
        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        # torch.nn.Embedding을 사용했습니다.
        self.embedding = Embedding(num_nodes, embedding_dim)
        # self.convs: 이 부분은 파라미터 없이 작동하고, 별도로 설정할 것이 없기에 웬만하면 건들지 않는게 좋습니다.
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        # weight의 초기값을 별도로 구성했습니다.
        self.reset_parameters()

    # predict_link()하는 부분에서 sigmoid를 사용하기에, xavier_uniform_으로 초기화하는 것이 좋습니다.
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    """
        edge_index:
            from torch_geometric.typing import Adj, OptTensor 를 보니, custom 타입임을 알 수 있습니다.
            실제 사용은, forward에서 train_data["edge"]를 사용합니다.
            train_data_proc: {"edge":[유저,문제], "label":문제 맞췄는지} 형태 참고
            n by 2 행렬
    """
    def get_embedding(self, edge_index: Adj) -> Tensor:
        # 아래 두 줄은 초기에 embedding을 통과하는 부분입니다.
        breakpoint()
        x = self.embedding.weight
        # alpha를 별도로 지정하지 않았으면, self.alpha[i]는 모두 같은 값을 가집니다.
        out = x * self.alpha[0]

        # layer들을 통과하는 부분입니다.
        # 정성적으로 이해하면, next 주변 노드의 정보를 가져가는 부분이라고 보면 됩니다.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out

    """
        edge_index:
            from torch_geometric.typing import Adj, OptTensor 를 보니, custom 타입임을 알 수 있습니다.
            실제 사용은, forward에서 train_data["edge"]를 사용합니다.
            train_data_proc: {"edge":[유저,문제], "label":문제 맞췄는지} 형태 참고
            n by 2 행렬
        models.py의 코드:
            # forward
            pred = model(train_data["edge"])
            loss = model.link_pred_loss(pred, train_data["label"])
    """
    def forward(self, edge_index: Adj,
                edge_label_index: OptTensor = None) -> Tensor:
        r"""Computes rankings for pairs of nodes.
        Args:
            edge_index (Tensor or SparseTensor): Edge tensor specifying the
                connectivity of the graph.
            edge_label_index (Tensor, optional): Edge tensor specifying the
                node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
        """
        if edge_label_index is None:
            if isinstance(edge_index, SparseTensor):
                edge_label_index = torch.stack(edge_index.coo()[:2], dim=0)
            else:
                edge_label_index = edge_index
        
        # k-hop layer 통과
        out = self.get_embedding(edge_index)
        """
            ★★★★★ 중요! ★★★★★
            out_src = out[edge_label_index[0]] 뜻:
                userId의 embedding vector이자, 일종의 latent vector이다.
                형태: torch.Size([userId 개수, embedding_dim])
                이 out_src 부분으로 할 수 있는 것:
                    이대로 train_data.csv에 "userId"로 groupby해서 이어붙일 수 있음.
            out_dst = out[edge_label_index[1]] 뜻:
                assessmentItemID의 embedding vector이자, 일종의 latent vector이다.
                형태: torch.Size([assessmentItemID 개수, embedding_dim])
                이 out_src 부분으로 할 수 있는 것:
                    이대로 train_data.csv에 "assessmentItemID"로 groupby해서 이어붙일 수 있음.
            뿐만 아니라, tag와 같이 여러 feature를 기준으로 embedding을 구할 수 있음.
            따라서, sequence 모델에 연속형 변수로 활용이 가능하다.
        """
        # edge_label_index.shape: 
        #                   torch.Size([2, 495046]) == torch.Size([2, 모든 interaction]) -> 2: userId 칼럼, assessmentItemID 칼럼
        # out.shape:
        #           torch.Size([16896, 64]) == torch.Size([userId 개수 + assessmentItemID 개수, embedding_dim])
        # out_src.shape:
        #               torch.Size([495046, 64]) == 각 interaction의 user 기준으로의 embedding vector == torch.Size([모든 interaction, embedding_dim])
        out_src = out[edge_label_index[0]]
        # out_dst.shape:
        #               torch.Size([495046, 64]) == 각 interaction의 assessmentItem 기준으로의 embedding vector == torch.Size([모든 interaction, embedding_dim])
        out_dst = out[edge_label_index[1]]
        return (out_src * out_dst).sum(dim=-1)

    def predict_link(self, edge_index: Adj, edge_label_index: OptTensor = None,
                     prob: bool = False) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.
        Args:
            prob (bool): Whether probabilities should be returned. (default:
                :obj:`False`)
        """
        pred = self(edge_index, edge_label_index).sigmoid()
        # 예측값을 반올림할 것인지 말것인지
        # classification할 것인지, regression할 것인지
        return pred if prob else pred.round()

    def recommend(self, edge_index: Adj, src_index: OptTensor = None,
                  dst_index: OptTensor = None, k: int = 1) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.
        Args:
            src_index (Tensor, optional): Node indices for which
                recommendations should be generated.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            dst_index (Tensor, optional): Node indices which represent the
                possible recommendation choices.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            k (int, optional): Number of recommendations. (default: :obj:`1`)
        """
        out_src = out_dst = self.get_embedding(edge_index)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index

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

    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor,
                            lambda_reg: float = 1e-4, **kwargs) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.
        .. note::
            The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
            in the :obj:`neg_edge_rank` entry must correspond to ranks of
            positive and negative edges of the same entity (*e.g.*, user).
        Args:
            pos_edge_rank (Tensor): Positive edge rankings.
            neg_edge_rank (Tensor): Negative edge rankings.
            lambda_reg (int, optional): The :math:`L_2` regularization strength
                of the Bayesian Personalized Ranking (BPR) loss.
                (default: 1e-4)
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
                function.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        return loss_fn(pos_edge_rank, neg_edge_rank, self.embedding.weight)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.
    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).
    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2
    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.
    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs) -> None:
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.
        .. note::
            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.
        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).mean()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs