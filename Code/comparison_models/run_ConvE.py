import os
import argparse
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_normal_
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

from experiment_utils import resolve_run_dir, set_random_seed, update_timing, write_json


class Args:
    def __init__(self, parsed_args=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.bs = 1024
        self.epochs = 25
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
        self.seed = None
        self.deterministic = False
        self.dataset_name = "Eedi"
        self.run_root = None
        self.run_id = None
        self.timing_file = None
        self.metrics_file = None
        self.negative_ratio = 5

        if parsed_args is not None:
            for key, value in vars(parsed_args).items():
                if value is not None:
                    setattr(self, key, value)

        if self.run_root and parsed_args is not None and parsed_args.save_path is None:
            self.save_path = str(
                resolve_run_dir(self.run_root, self.dataset_name, "ConvE", run_id=self.run_id, seed=self.seed)
            )
        self.negative_ratio = max(0, int(self.negative_ratio))
        self.timing_file = self.timing_file or os.path.join(self.save_path, "timing.json")
        self.metrics_file = self.metrics_file or os.path.join(self.save_path, "metrics.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train ConvE for ER recommendation triples.")
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--cuda", type=lambda value: str(value).lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--run_root", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--timing_file", default=None)
    parser.add_argument("--metrics_file", default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--embedding_shape1", type=int, default=None)
    parser.add_argument("--input_drop", type=float, default=None)
    parser.add_argument("--hidden_drop", type=float, default=None)
    parser.add_argument("--feat_drop", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--negative-ratio", "--negative_ratio", dest="negative_ratio", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume ConvE training from last.pt in save_path.")
    return parser.parse_args(argv)


def training_triple_files(args):
    return ["triples.txt"]


def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def entity_type(entity_name):
    if entity_name.startswith("uid"):
        return "uid"
    if entity_name.startswith("ex"):
        return "ex"
    if entity_name.startswith("kc"):
        return "kc"
    return None


def relation_tail_type(relation_name):
    if relation_name == "rec":
        return "ex"
    if relation_name.startswith(("mlkc", "pkc", "exfr")):
        return "uid"
    return None


def relation_uses_negative_sampling(relation_name):
    return relation_name == "rec"


def build_tail_candidates(entity2id):
    candidates = {"uid": [], "ex": [], "kc": []}
    for entity_name, entity_id in entity2id.items():
        kind = entity_type(entity_name)
        if kind in candidates:
            candidates[kind].append(entity_id)
    return {key: sorted(value) for key, value in candidates.items()}


def build_positive_tails_by_hr(triples):
    positive_tails = {}
    for h, r, t in triples:
        positive_tails.setdefault((h, r), set()).add(t)
    return positive_tails


class MyDataset(Dataset):
    def __init__(
        self,
        triples,
        entity2id,
        relation2id,
        cuda=True,
        negative_ratio=0,
        relation_id_to_name=None,
        entity_id_to_name=None,
        tail_candidates_by_type=None,
        positive_tails_by_hr=None,
        seed=None,
    ):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.cuda = cuda
        self.negative_ratio = max(0, int(negative_ratio or 0))
        self.relation_id_to_name = relation_id_to_name or {}
        self.entity_id_to_name = entity_id_to_name or {}
        self.tail_candidates_by_type = tail_candidates_by_type or {}
        self.positive_tails_by_hr = positive_tails_by_hr or {}
        self.seed = 0 if seed is None else int(seed)
        self.sample_index = self.build_sample_index()

    def should_sample_negative(self, relation_id):
        relation_name = self.relation_id_to_name.get(relation_id, "")
        return relation_uses_negative_sampling(relation_name) or relation_id == self.relation2id.get("rec")

    def build_sample_index(self):
        sample_index = []
        for triple_idx, (_, relation_id, _) in enumerate(self.triples):
            sample_index.append((triple_idx, 0))
            if self.should_sample_negative(relation_id):
                for negative_idx in range(1, self.negative_ratio + 1):
                    sample_index.append((triple_idx, negative_idx))
        return sample_index

    def __len__(self):
        return len(self.sample_index)

    def sample_negative_tail(self, h, r, positive_tail, sample_idx):
        relation_name = self.relation_id_to_name.get(r, "")
        tail_type = relation_tail_type(relation_name) or entity_type(self.entity_id_to_name.get(positive_tail, ""))
        candidates = self.tail_candidates_by_type.get(tail_type, [])
        if not candidates:
            raise ValueError(f"No type-compatible tail candidates for relation {relation_name}")

        blocked = self.positive_tails_by_hr.get((h, r), set())
        rng = random.Random(self.seed + sample_idx * 1000003)
        for _ in range(100):
            candidate = candidates[rng.randrange(len(candidates))]
            if candidate not in blocked:
                return candidate

        eligible = [candidate for candidate in candidates if candidate not in blocked]
        if not eligible:
            raise ValueError(f"No filtered negative tail available for relation {relation_name}")
        return eligible[rng.randrange(len(eligible))]

    def __getitem__(self, idx):
        triple_idx, offset = self.sample_index[idx]
        h, r, t = self.triples[triple_idx]
        label = 1.0
        if offset != 0:
            t = self.sample_negative_tail(h, r, t, idx)
            label = 0.0
        if self.cuda:
            return (
                torch.tensor(h).cuda(),
                torch.tensor(r).cuda(),
                torch.tensor(t).cuda(),
                torch.tensor(label, dtype=torch.float32).cuda(),
            )
        return torch.tensor(h), torch.tensor(r), torch.tensor(t), torch.tensor(label, dtype=torch.float32)


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

    def forward(self, h, r, t=None):
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

        # 与所有实体做乘积
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)

        pred_score = torch.sigmoid(x)

        if t is not None:
            tail_score = pred_score.gather(1, t.view(-1, 1))
            return tail_score
        return pred_score

    def inference(self, h_emb, r_emb, t_emb):
        stacked_inputs = torch.cat([h_emb.view(-1, 1, self.emb_dim1, self.emb_dim2),
                                    r_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)], 2)
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

        x = torch.mm(x, t_emb.transpose(1,0))
        x += self.b.expand_as(x)
        score = torch.sigmoid(x)
        return score


# 训练步骤
def train_step(model, h, r, t, label, optimizer, cuda=True):
    optimizer.zero_grad()
    pred = model.forward(h, r)
    pred_t = torch.gather(pred, 1, t.unsqueeze(1))  # 选取t的预测值
    label = label.to(pred_t.device).float().view(-1)
    loss = F.binary_cross_entropy(pred_t.view(-1), label)
    loss.backward()
    optimizer.step()
    return loss.item()

def inference(model, h, r, t):
    tail_score = model(h, r, t)
    return tail_score


def torch_save_file(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fp:
        torch.save(obj, fp)


def torch_load_file(path, map_location=None):
    with open(path, "rb") as fp:
        return torch.load(fp, map_location=map_location)


def save_model(model, optimizer, args, epoch=None, loss=None, checkpoint_name="last.pt", save_legacy=True):
    os.makedirs(args.save_path, exist_ok=True)
    payload = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch_save_file(payload, os.path.join(args.save_path, checkpoint_name))
    if save_legacy and checkpoint_name != "checkpoint":
        torch_save_file(payload, os.path.join(args.save_path, "checkpoint"))
    is_best = checkpoint_name == "best.pt"
    if is_best:
        torch_save_file(model, os.path.join(args.save_path, "best_model.pth"))
    else:
        torch_save_file(model, os.path.join(args.save_path, 'DTransformer.pth'))
        torch_save_file(model, os.path.join(args.save_path, 'model.pth'))

    with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    prefix = "best_" if is_best else ""
    entity_embedding = model.emb_e.weight.data.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, f'{prefix}entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.emb_rel.weight.data.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, f'{prefix}relation_embedding'),
        relation_embedding
    )


def load_training_checkpoint(model, optimizer, checkpoint_path, cuda=True):
    map_location = None if cuda else torch.device("cpu")
    checkpoint = torch_load_file(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch") or 0), checkpoint.get("loss")


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
    args = Args(parse_args())
    os.makedirs(args.save_path, exist_ok=True)
    set_random_seed(args.seed, deterministic=args.deterministic)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
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

    train_files = training_triple_files(args)
    entity_id_to_name = {entity_id: entity_name for entity_name, entity_id in entity2id.items()}
    relation_id_to_name = {relation_id: relation_name for relation_name, relation_id in relation2id.items()}
    tail_candidates_by_type = build_tail_candidates(entity2id)
    triple_groups = []
    positive_triples = []
    total_triples = 0
    for file_name in train_files:
        file_triples = read_triple(os.path.join(args.data_path, file_name), entity2id, relation2id)
        total_triples += len(file_triples)
        positive_triples.extend(file_triples)

    positive_tails_by_hr = build_positive_tails_by_hr(positive_triples)
    total_training_samples = 0
    for file_name in train_files:
        file_triples = read_triple(os.path.join(args.data_path, file_name), entity2id, relation2id)
        dataset = MyDataset(
            file_triples,
            entity2id,
            relation2id,
            args.cuda,
            negative_ratio=args.negative_ratio,
            relation_id_to_name=relation_id_to_name,
            entity_id_to_name=entity_id_to_name,
            tail_candidates_by_type=tail_candidates_by_type,
            positive_tails_by_hr=positive_tails_by_hr,
            seed=args.seed,
        )
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
        triple_groups.append((file_name, dataloader, len(file_triples)))
        total_training_samples += len(dataset)

    model = ConvE(args, nentity, nrelation)
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0
    if getattr(args, "resume", False):
        checkpoint_path = os.path.join(args.save_path, "last.pt")
        if os.path.exists(checkpoint_path):
            start_epoch, last_loss = load_training_checkpoint(model, optimizer, checkpoint_path, cuda=args.cuda)
            if last_loss is not None:
                best_loss = float(last_loss)
                best_epoch = start_epoch
            logging.info(f"Resumed ConvE from {checkpoint_path}, start_epoch={start_epoch}, last_loss={last_loss}")
            best_checkpoint_path = os.path.join(args.save_path, "best.pt")
            if os.path.exists(best_checkpoint_path):
                best_checkpoint = torch_load_file(best_checkpoint_path, map_location=None if args.cuda else torch.device("cpu"))
                if best_checkpoint.get("loss") is not None:
                    best_loss = float(best_checkpoint["loss"])
                    best_epoch = int(best_checkpoint.get("epoch") or best_epoch)
                logging.info(f"Existing best ConvE checkpoint: epoch={best_epoch}, loss={best_loss}")
        else:
            logging.info(f"--resume requested but {checkpoint_path} does not exist; starting from scratch.")

    # 训练过程
    logging.info('Start Training...')
    logging.info(f'batchsize = {args.bs}')
    logging.info(f'epochs = {args.epochs}')
    logging.info(f'learning_rate = {args.learning_rate}')
    logging.info(f'seed = {args.seed}')
    logging.info(f'input_drop = {args.input_drop}')
    logging.info(f'hidden_drop = {args.hidden_drop}')
    logging.info(f'feat_drop = {args.feat_drop}')
    logging.info(f'training_triple_files = {train_files}')
    logging.info(f'positive_train_triples = {total_triples}')
    logging.info(f'negative_ratio = {args.negative_ratio}')
    logging.info(f'total_training_samples = {total_training_samples}')

    training_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0
        batch_count = 0
        for file_name, dataloader, _ in triple_groups:
            logging.info(f"Epoch {epoch + 1}: training on {file_name}")
            for h, r, t, label in tqdm(dataloader):
                loss = train_step(model, h, r, t, label, optimizer, args.cuda)
                epoch_loss += loss
                batch_count += 1

        avg_loss = epoch_loss / max(1, batch_count)
        save_model(model, optimizer, args, epoch=epoch + 1, loss=avg_loss, checkpoint_name="last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            save_model(model, optimizer, args, epoch=epoch + 1, loss=avg_loss, checkpoint_name="best.pt", save_legacy=False)

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss}")
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss}")

    training_seconds = time.perf_counter() - training_start
    update_timing(args.timing_file, "training", training_seconds, extra={"epochs": args.epochs})
    write_json(
        {
            "model": "ConvE",
            "dataset": args.dataset_name,
            "seed": args.seed,
            "train_loss_best": best_loss if best_loss < float("inf") else None,
            "best_epoch": best_epoch,
            "final_epoch": args.epochs,
            "training_seconds": round(training_seconds, 6),
            "input_drop": args.input_drop,
            "hidden_drop": args.hidden_drop,
            "feat_drop": args.feat_drop,
            "training_triple_files": train_files,
            "positive_train_triples": total_triples,
            "negative_ratio": args.negative_ratio,
            "total_training_samples": total_training_samples,
            "best_model_file": "best_model.pth",
        },
        args.metrics_file,
    )



if __name__ == "__main__":
    # torch.cuda.empty_cache()
    main()
