import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (BertConfig,
                                                        BertEncoder, BertModel)


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        # self.embedding_list = [
        #     nn.Embedding(input_size + 1, self.hidden_dim // 3, device=args.device)
        #     for input_size in self.args.embed_layer_input_size_list
        # ]
        # line 27에서, nn.Embedding에 device를 넣어줘야만 에러가 해결되는 현상이 있음.
        # 에러 해결을 위해 torch 버전을 바꾸면, cuda가 사용 불가능한 상황
        # 정확한 원인이나 해결책은 모르겠음. 버그가 아닌가 싶기도 함
        # 일단은 nn.Embedding에 device를 추가적으로 지정해주니 에러가 해결됨
        # 발생했던 에러:
        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper__index_select)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)
        # self.comb_proj = nn.Linear(
        #     (self.hidden_dim // 3) * (1 + len(self.embedding_list)), self.hidden_dim
        # )

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):
        # input[3]: correct, input[-1]: interaction, input[-2]: mask

        # test, question, tag, _, mask, interaction = input
        _, test, question, tag, mask, interaction = input
        # _, test, question, tag, mask, interaction = input

        batch_size = interaction.size(0)
        # batch_size = input[-1].size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        # embed_interaction = self.embedding_interaction(input[-1])
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        # embed_list = [
        #     self.embedding_list[i](input[i + 1])
        #     for i in range(len(self.embedding_list))
        # ]

        embed = torch.cat(
            [
                embed_test,
                embed_question,
                embed_tag,
                embed_interaction,
            ],
            2,
        )
        # embed = torch.cat(
        #     [embed_interaction] + embed_list,
        #     2,
        # )

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        # self.embedding_list = [
        #     nn.Embedding(input_size + 1, self.hidden_dim // 3, device=args.device)
        #     for input_size in self.args.embed_layer_input_size_list
        # ]

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)
        # self.comb_proj = nn.Linear(
        #     (self.hidden_dim // 3) * (1 + len(self.embedding_list)), self.hidden_dim
        # )

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # input[3]: correct, input[-1]: interaction, input[-2]: mask

        # test, question, tag, _, mask, interaction = input
        _, test, question, tag, mask, interaction = input
        # _, test, question, tag, mask, interaction = input

        batch_size = interaction.size(0)
        # batch_size = input[-1].size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        # embed_interaction = self.embedding_interaction(input[-1])
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        # embed_list = [
        #     self.embedding_list[i](input[i + 1])
        #     for i in range(len(self.embedding_list))
        # ]

        embed = torch.cat(
            [
                embed_test,
                embed_question,
                embed_tag,
                embed_interaction,
            ],
            2,
        )
        # embed = torch.cat(
        #     [embed_interaction] + embed_list,
        #     2,
        # )

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = input[-2].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        # self.embedding_list = [
        #     nn.Embedding(input_size + 1, self.hidden_dim // 3, device=args.device)
        #     for input_size in self.args.embed_layer_input_size_list
        # ]

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)
        # self.comb_proj = nn.Linear(
        #     (self.hidden_dim // 3) * (1 + len(self.embedding_list)), self.hidden_dim
        # )

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # input[3]: correct, input[-1]: interaction, input[-2]: mask

        # test, question, tag, _, mask, interaction = input
        _, test, question, tag, mask, interaction = input
        # _, test, question, tag, mask, interaction = input

        batch_size = interaction.size(0)
        # batch_size = input[-1].size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        # embed_interaction = self.embedding_interaction(input[-1])
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        # embed_list = [
        #     self.embedding_list[i](input[i + 1])
        #     for i in range(len(self.embedding_list))
        # ]

        embed = torch.cat(
            [
                embed_test,
                embed_question,
                embed_tag,
                embed_interaction,
            ],
            2,
        )
        # embed = torch.cat(
        #     [embed_interaction] + embed_list,
        #     2,
        # )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        # encoded_layers = self.encoder(inputs_embeds=X, attention_mask=input[-2])
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out).view(batch_size, -1)
        return out


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # Encoder
        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.args.n_heads
        )
        self.mask = None  # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        _, test, question, tag, mask, interaction = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################

        q = self.query(embed).permute(1, 0, 2)

        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)

        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # out = self.fc(out)

        # preds = self.activation(out).view(batch_size, -1)
        preds = self.fc(out).view(batch_size, -1)

        return preds
