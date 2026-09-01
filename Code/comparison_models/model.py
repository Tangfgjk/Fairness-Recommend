#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


"""
知识图谱嵌入模型（Knowledge Graph Embedding, KGE），主要实现 TransE 和 RotatE 两种方法。

两种方法：把 知识图谱中的实体和关系 映射到低维向量空间中，用向量运算来表示 (头实体, 关系, 尾实体) 这种三元组的语义。
"""
class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, triplere_u,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.u = triplere_u

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        #Do not forget to modify this line when you add a new DTransformer in the "forward" function
        if model_name not in ['TransE', 'RotatE', 'DistMult', 'ComplEx']:
            raise ValueError('DTransformer %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'RotatE': self.RotatE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('DTransformer %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        # head: [bs, 1, hidden_dim*2]
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def DistMult(self, head, relation, tail, mode):
        score = head * relation * tail
        return score.sum(dim=2)

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        return score.sum(dim=2)



    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()

        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 +
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

"""
ConvE 类 （卷积模型）:利用卷积提取实体 + 关系的交互特征
"""
class ConvE(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, num_filters=32):
        super(ConvE, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim  # 输出特征维度
        self.num_filters = num_filters

        self.input_shape = (25, 20)
        # self.embedding_range = nn.Parameter(
        #     torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
        #     requires_grad=False
        # )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))

        nn.init.xavier_uniform_(self.entity_embedding)
        nn.init.xavier_uniform_(self.relation_embedding)
        # nn.init.uniform_(
        #     tensor=self.entity_embedding,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )
        # nn.init.uniform_(
        #     tensor=self.relation_embedding,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )


        # 卷积层
        self.conv_layer = nn.Conv2d(2, num_filters, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(0.1)

        # 全连接层，用于将卷积后的特征展平
        if self.entity_dim == self.relation_dim:
            flattened_size = num_filters * self.input_shape[0] * self.input_shape[1]  #self.entity_dim
            self.fc = nn.Linear(flattened_size, hidden_dim)
        else:
            raise ValueError("注意尺寸")

    def inference_tail(self, head_idx, rela_idx):
        h = self.entity_embedding[head_idx]
        r = self.relation_embedding[rela_idx]

        batch_size = head_idx.size(0)

        h_reshaped = h.view(batch_size, 1, *self.input_shape)
        r_reshaped = r.view(batch_size, 1, *self.input_shape)
        combined = torch.cat([h_reshaped, r_reshaped], dim=1)

        conv_out = F.relu(self.conv_layer(combined))
        conv_out = self.dropout(conv_out)

        conv_out_flattened = conv_out.view(batch_size, -1)
        feature_vec = F.relu(self.fc(conv_out_flattened))
        return feature_vec

    def forward(self, sample, mode='single'):
        h, r = None, None
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            h = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            h = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        h_reshaped = h.view(batch_size, 1, *self.input_shape)
        r_reshaped = r.view(batch_size, 1, *self.input_shape)
        combined = torch.cat([h_reshaped, r_reshaped], dim=1)

        # 卷积
        conv_out = F.relu(self.conv_layer(combined))
        conv_out = self.dropout(conv_out)

        # 展平
        conv_out_flattened = conv_out.view(batch_size, -1)
        feature_vec = F.relu(self.fc(conv_out_flattened))

        if mode == 'single':
            scores = torch.sum(feature_vec * t, dim=1)
            # t = t.squeeze(1)  # 去掉多余的维度
            # scores = torch.sum(feature_vec * t, dim=1)
        elif mode == 'tail-batch':
            # 负例情况下，扩展 feature_vec 的维度
            feature_vec = feature_vec.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            scores = torch.sum(feature_vec * t, dim=2)  # 计算分数
        return scores

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):

        model.train()
        optimizer.zero_grad()

        positive_sample, _, _, mode = next(train_iterator)
        positive_sample = positive_sample.cuda()


        loss = (positive_sample_loss + negative_sample_loss) / 2

        # if args.regularization != 0.0:
        #     # Use L3 regularization for ComplEx and DistMult
        #     regularization = args.regularization * (
        #             DTransformer.entity_embedding.norm(p=3) ** 3 +
        #             DTransformer.relation_embedding.norm(p=3).norm(p=3) ** 3
        #     )
        #     loss = loss + regularization
        #     regularization_log = {'regularization': regularization.item()}
        # else:
        #     regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            # **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
