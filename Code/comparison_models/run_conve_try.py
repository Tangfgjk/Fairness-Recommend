import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal_
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

class Args:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.bs = 1024
        self.epochs = 10
        self.learning_rate = 0.001
        self.cuda = True
        self.data_path = os.path.abspath(os.path.join(base_dir, '..', 'data', 'Eedi'))
        self.save_path = os.path.join(base_dir, 'models', 'Eedi', 'ConvE')

        self.embedding_dim = 200
        self.embedding_shape1 = 20
        self.input_drop = 0.2
        self.hidden_drop = 0.2
        self.feat_drop = 0.3
        self.hidden_size = 9728  # fc隐藏层大小
        self.use_bias = True
        # self.data_path = '../data/Eedi'


def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


class MyDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id, cuda=True):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.cuda = cuda

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        if self.cuda:
            return torch.tensor(h).cuda(), torch.tensor(r).cuda(), torch.tensor(t).cuda()
        return torch.tensor(h), torch.tensor(r), torch.tensor(t)


# 定义模型（以ConvE为例）
class ConvE(nn.Module):
    def __init__(self, args, nentity, nrelation):
        super(ConvE, self).__init__()
        self.emb_e = nn.Embedding(nentity, args.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(nrelation, args.embedding_dim, padding_idx=0)
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.inp_drop = nn.Dropout(args.input_drop)
        self.hidden_drop = nn.Dropout(args.hidden_drop)
        self.feature_map_drop = nn.Dropout2d(args.feat_drop)

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(args.embedding_dim)
        self.fc = nn.Linear(args.hidden_size, args.embedding_dim)

        self.register_parameter('b', nn.Parameter(torch.zeros(nentity)))
        self.init()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, h, r):
        h_emb = self.emb_e(h).view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_emb = self.emb_rel(r).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([h_emb, r_emb], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)

        pred = torch.sigmoid(x)
        return pred


# 训练步骤
def train_step(model, h, r, t, label, optimizer, cuda=True):
    optimizer.zero_grad()
    pred = model.forward(h, r)
    pred_t = torch.gather(pred, 1, t.unsqueeze(1))  # 选取t的预测值
    if cuda:
        label = label.cuda()
        pred_t = pred_t.cuda()
    loss = F.binary_cross_entropy(pred_t.squeeze(), label.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def torch_save_file(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        torch.save(obj, fp)


def save_embedding(model, optimizer, args, save_model=False):
    if save_model:
        torch_save_file({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(args.save_path, 'checkpoint')
        )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )

def set_logger(args):

    log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    
def main():
    args = Args()
    set_logger(args)
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    triples = read_triple(os.path.join(args.data_path, 'triples.txt'), entity2id, relation2id)

    dataset = MyDataset(triples, entity2id, relation2id, args.cuda)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    model = ConvE(args, nentity, nrelation)
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练过程

    logging.info('Start Training...')
    logging.info('batchsize = %d' % args.bs)
    logging.info('epochs = %d' % args.epochs)
    logging.info('learning_rate = %d' % args.learning_rate)

    for epoch in range(args.epochs):
        epoch_loss = 0
        for h, r, t in tqdm(dataloader):
            label = torch.tensor([1] * len(h))  # 假设所有三元组为正确，实际可以使用负采样等方式
            loss = train_step(model, h, r, t, label, optimizer, args.cuda)
            epoch_loss += loss

        save_embedding(model, optimizer, args)

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(dataloader)}")
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(dataloader)}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
