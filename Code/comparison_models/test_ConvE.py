import os
import pickle
import argparse
import sys
import json
try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import time
from datetime import datetime
from explain_recommendations import generate_explanation_cards, write_explanation_cards

# 记录操作开始时间
start_time = datetime.now()
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

    def inference(self, h_emb, r_emb, t_emb, index_list=None):
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

        x = torch.mm(x, t_emb.transpose(1, 0))  # [234, 948]
        if index_list is not None:
            x += torch.index_select(self.b, 0, torch.tensor(index_list)).unsqueeze(-1).expand_as(x)  # cuda()
        else:
            x += self.b.expand_as(x)
        score = torch.sigmoid(x)
        return score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvE recommendations and export explanation cards.")
    parser.add_argument("--dataset", default="Eedi")
    parser.add_argument("--model-type", default="ConvE")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--embedding-path", default=None)
    parser.add_argument("--explain-top-k", type=int, default=3)
    parser.add_argument("--explain-user-count", type=int, default=2)
    parser.add_argument("--explain-users", default="", help="Comma-separated uid list, for example uid0,uid1.")
    parser.add_argument("--explain-output-dir", default=None)
    parser.add_argument("--skip-explanations", action="store_true")
    parser.add_argument("--timing-file", default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Only export recommendation scores and explanation cards. Metric calculation is handled by evaluate_recommendations.py.",
    )
    return parser.parse_args()


def update_timing(timing_file, stage, seconds):
    if not timing_file:
        return
    payload = {}
    if os.path.exists(timing_file):
        with open(timing_file, "r", encoding="utf-8") as fp:
            try:
                payload = json.load(fp)
            except json.JSONDecodeError:
                payload = {}
    payload[stage] = {"seconds": round(seconds, 6)}
    os.makedirs(os.path.dirname(timing_file), exist_ok=True)
    with open(timing_file, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def torch_load_file(path, map_location=None):
    with open(path, "rb") as fp:
        return torch.load(fp, map_location=map_location)


def resolve_conve_artifacts(embedding_path, checkpoint="best"):
    if checkpoint == "best":
        best_model = os.path.join(embedding_path, "best_model.pth")
        best_entity = os.path.join(embedding_path, "best_entity_embedding.npy")
        best_relation = os.path.join(embedding_path, "best_relation_embedding.npy")
        if os.path.exists(best_model) and os.path.exists(best_entity) and os.path.exists(best_relation):
            return best_model, best_entity, best_relation, "best"
        print("Best ConvE artifacts not found; falling back to last model artifacts.")

    return (
        os.path.join(embedding_path, "model.pth"),
        os.path.join(embedding_path, "entity_embedding.npy"),
        os.path.join(embedding_path, "relation_embedding.npy"),
        "last",
    )


args = parse_args()
dataset = args.dataset
type = args.model_type

dict_path = args.data_path or f"../data/{dataset}"
embedding_path = args.embedding_path or f"./models/{dataset}/{type}"
model_file, entity_embedding_file, relation_embedding_file, checkpoint_used = resolve_conve_artifacts(
    embedding_path,
    checkpoint=args.checkpoint,
)

relation_embedding = np.load(relation_embedding_file)
entity_embedding = np.load(entity_embedding_file)
# 将自定义类加入白名单
if add_safe_globals is not None:
    add_safe_globals([
        ConvE, nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d,
        nn.Conv2d, nn.Linear, nn.Dropout, nn.Dropout2d
    ])
convE_model = torch_load_file(model_file, map_location=torch.device('cpu'))
convE_model.eval()
print(f"Loaded ConvE checkpoint: {checkpoint_used} ({model_file})")
# convE_model = convE_model.cuda()

### read Q-matrix, entities, relations
Q = []
with open(f"{dict_path}/Q.txt", 'r') as file:
    i = 0
    for line in file:
        kc = line.strip().split(',')
        kc_int = [int(x) for x in kc]
        Q.append(kc_int)

with open(f"{dict_path}/entities.dict", 'r') as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(f"{dict_path}/relations.dict", 'r') as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


### read embeddings
dict_entity_embedding = {}
dict_relation_embedding = {}

for (k, v) in entity2id.items():
    dict_entity_embedding[k] = entity_embedding[v, :]

for (k, v) in relation2id.items():
    dict_relation_embedding[k] = relation_embedding[v, :]


### read test_triples
uid_mlkc_dict = {}
uid_pkc_dict = {}
uid_exfr_dict = {}
uid_rec_ex_dict = {}

with open(f"{dict_path}/test_triples.txt", 'r', encoding="UTF-8") as load_file:
    for line in load_file:
        item1, item2, uid = line.strip().split('\t')
        if item2[0] == 'm':
            kc, mlkc, uid = item1, item2, uid
            if uid not in uid_mlkc_dict.keys():
                uid_mlkc_dict[uid] = {}
            uid_mlkc_dict[uid][kc] = 'mlkc' + str(round(float(mlkc[4:]), 2))
        elif item2[0] == 'e':
            ex, exfr, uid = item1, item2, uid
            if uid not in uid_exfr_dict.keys():
                uid_exfr_dict[uid] = {}
            uid_exfr_dict[uid][ex] = 'exfr' + str(round(float(exfr[4:]), 2))
        else:
            kc, pkc, uid = item1, item2, uid
            if uid not in uid_pkc_dict.keys():
                uid_pkc_dict[uid] = {}
            uid_pkc_dict[uid][kc] = 'pkc' + str(round(float(pkc[3:]), 2))


### calculate the score of each user for each exercise
uid_ex_scores = []
user_num = 0
rec_embedding = torch.from_numpy(dict_relation_embedding['rec'])#.cuda()
uid_mlkc_dict_keys_list = [key for key in uid_mlkc_dict.keys()]
print("start!!!")
inference_start = time.perf_counter()

if True:#not os.path.exists(f'{embedding_path}/{type}_uid_ex_scores.pkl'):  # 如果文件不存在
    stu_idx_list = [int(uid[3:]) for uid in uid_mlkc_dict_keys_list]
    stu_list = [torch.from_numpy(dict_entity_embedding[uid]) for uid in uid_mlkc_dict_keys_list]
    exe_list = [torch.from_numpy(dict_entity_embedding['ex' + str(qid)]) for qid in range(len(Q))]

    stu_emb = torch.stack(stu_list, dim=0)#.cuda()  # [234, 200]
    rec_emb = rec_embedding.unsqueeze(0).repeat(stu_emb.shape[0], 1)#.cuda()
    exe_emb = torch.stack(exe_list, dim=0)#.cuda()  # [948, 200]

    rec_score = convE_model.inference(stu_emb, rec_emb, exe_emb, stu_idx_list)
    # rec_score = convE_model(stu_emb, rec_emb)

    for id, rec in enumerate(rec_score):
        scores = [s.item() for s in rec]
        uid_ex_scores.append((f"uid{stu_idx_list[id]}", scores))
    with open(f'{embedding_path}/{type}_uid_ex_scores.pkl', 'wb') as f:
        pickle.dump(uid_ex_scores, f)
else:
    with open(f'{embedding_path}/{type}_uid_ex_scores.pkl', 'rb') as f:
        uid_ex_scores = pickle.load(f)
    print("文件已存在，读取内容")

update_timing(args.timing_file, "inference_without_cache", time.perf_counter() - inference_start)

if not args.skip_explanations:
    selected_users = [uid.strip() for uid in args.explain_users.split(",") if uid.strip()] or None
    explanation_output_dir = args.explain_output_dir or os.path.join(embedding_path, "explanations")
    explanation_cards = generate_explanation_cards(
        uid_ex_scores=uid_ex_scores,
        q_matrix=Q,
        uid_mlkc_dict=uid_mlkc_dict,
        uid_pkc_dict=uid_pkc_dict,
        uid_exfr_dict=uid_exfr_dict,
        top_k=args.explain_top_k,
        user_limit=args.explain_user_count,
        selected_users=selected_users,
        model_name=type,
    )
    json_path, md_path = write_explanation_cards(
        explanation_cards,
        output_dir=explanation_output_dir,
        file_prefix=f"{type}_top{args.explain_top_k}_explanations",
    )
    print(f"Explanation cards saved to: {json_path}")
    print(f"Explanation cards saved to: {md_path}")

if args.scores_only:
    sys.exit(0)

def ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n):
    acc = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]

        user_mlkc = uid_mlkc_dict[uid]
        diff = 0
        for ex_id in uid_ex_score:
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            ex_ml = 1.0
            for kc in kc_list:
                ex_ml = ex_ml * float(user_mlkc['kc' + str(kc)][4:])
            diff += 1 - np.abs(r1 - (ex_ml))
        acc.append(diff / n)
    return np.mean(acc), np.std(acc)

def Nov(uid_kc_response, uid_ex_scores, Q, n):
    jaccsim = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        kc_response = set(uid_kc_response[uid])
        jaccard_similarity = 0
        for ex_id in uid_ex_score:
            rec_ex_kc_set = set()
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            rec_ex_kc_set.update(kc_list)
            intersection = len(kc_response.intersection(rec_ex_kc_set))
            union = len(kc_response.union(rec_ex_kc_set))
            jaccard_similarity += 1 - intersection / union
        jaccsim.append(jaccard_similarity / n)
    return np.mean(jaccsim), np.std(jaccsim)


def TimeWeighted_Nov_with_Last(uid_kc_response, uid_ex_scores, Q, n, alpha=0.1):
    """
    基于时间加权的新颖度计算，将最后一个时刻 t+1 作为基准。
    :param uid_kc_response: 学生的作答知识点记录 {uid: [kc1, kc2, ...]}，按时间顺序排列
    :param uid_ex_scores: 推荐试题得分 {uid: [(ex_id, score), ...]}
    :param Q: 知识点矩阵 (questions x knowledge points)，Q[i][j] = 1 表示试题 i 涉及知识点 j
    :param n: 推荐试题数目
    :param alpha: 时间衰减参数
    :return: 新颖度的均值和标准差
    """
    jaccsim = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        kc_response = uid_kc_response[uid]

        # 当前时刻 t+1
        t_plus_1 = len(kc_response)

        # 知识点最后作答时间字典
        kc_last_time = {kc: t for t, kc in enumerate(kc_response, 1)}

        jaccard_similarity = 0
        for ex_id in uid_ex_score:
            rec_ex_kc_set = set()
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            rec_ex_kc_set.update(kc_list)

            intersection = set(kc_response).intersection(rec_ex_kc_set)
            union = set(kc_response).union(rec_ex_kc_set)

            # 计算时间加权
            weighted_intersection = sum(
                np.exp(-alpha * (t_plus_1 - kc_last_time.get(kc, t_plus_1 + 1))) for kc in intersection
            )
            weighted_union = sum(
                np.exp(-alpha * (t_plus_1 - kc_last_time.get(kc, t_plus_1 + 1))) for kc in union
            )
            time_weighted_jaccard = 1 - weighted_intersection / weighted_union
            jaccard_similarity += time_weighted_jaccard
        jaccsim.append(jaccard_similarity / n)
    return np.mean(jaccsim), np.std(jaccsim)


end_time = datetime.now()
# 计算时间间隔
time_difference = end_time - start_time
print(f"时间间隔（秒）: {time_difference.total_seconds()} 秒")


def find_exer(uid_ex_score, n):
    uid, scores = uid_ex_score[0], uid_ex_score[1]
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    score = [item[0] for item in sorted_scores][:n]
    print(score)


item_idx = 1
find_exer(uid_ex_scores[item_idx], 30)


print("-----------------------------------------------Start calculating ACC-----------------------------------------------")
r1 = 0.8
acc_dic = {}
# for n in [10, 15, 20, 30, 50, 75, 100]:
for n in range(10,101,10):
    mean_acc, std_acc = ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n)
    print(f"The recommendation list length is n = {n}, the mean ACC = {mean_acc}, the std ACC = {std_acc}")
    acc_dic[n] = mean_acc
mean_acc, std_acc = ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, 15)
print(f"The recommendation list length is n = 15, the mean ACC = {mean_acc}, the std ACC = {std_acc}")
mean_acc, std_acc = ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, 75)
print(f"The recommendation list length is n = 75, the mean ACC = {mean_acc}, the std ACC = {std_acc}")
#
# print("-----------------------------------------------Start calculating NOV-----------------------------------------------")
all_uid_kc_response = {}
with open(f"{dict_path}/{dataset}_uid_kc_response.txt", 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        uid = line[0]
        correct_kc_response = [int(x) for x in line[1].split(',')]
        all_uid_kc_response[uid] = correct_kc_response

test_uid_kc_response = {}
for uid in uid_mlkc_dict.keys():
    test_uid_kc_response[uid] = all_uid_kc_response[uid]


# for n in [10, 15, 20, 30, 40, 50, 75, 100]:
nov_dic = {}
for n in range(10,101,10):
    # mean_nov, std_nov = Nov(test_uid_kc_response, uid_ex_scores, Q, n)
    mean_nov, std_nov = TimeWeighted_Nov_with_Last(test_uid_kc_response, uid_ex_scores, Q, n, 1)
    print(f"The recommendation list length is n = {n}, the mean NOV = {mean_nov}, the std NOV = {std_nov}")
    nov_dic[n] = mean_nov
mean_nov, std_nov = TimeWeighted_Nov_with_Last(test_uid_kc_response, uid_ex_scores, Q, 15, 1)
print(f"The recommendation list length is n = 15, the mean NOV = {mean_nov}, the std NOV = {std_nov}")
mean_nov, std_nov = TimeWeighted_Nov_with_Last(test_uid_kc_response, uid_ex_scores, Q, 75, 1)
print(f"The recommendation list length is n = 75, the mean NOV = {mean_nov}, the std NOV = {std_nov}")
#
#
# # def create_excel(acc_dic, t="acc"):
# #     # 将字典转换为DataFrame
# #     df = pd.DataFrame(list(acc_dic.items()), columns=['n', t])
# #
# #     # 将DataFrame写入Excel文件
# #     excel_path = 'output.xlsx'
# #     df.to_excel(excel_path, index=False)
# #
# #     # 可选：使用openpyxl或xlsxwriter来创建一个带有折线图的Excel文件
# #     # 这里我们使用openpyxl作为示例
# #     from openpyxl import Workbook
# #     from openpyxl.chart import LineChart, Reference
# #
# #     # 读取刚才写入的Excel文件
# #     wb = Workbook()
# #     ws = wb.active
# #
# #     # 由于我们已经将数据写入了一个Excel文件，这里我们再次读取它以便在同一个文件中添加图表
# #     # 但为了示例简洁，我们直接从这里重新写入数据到新的工作簿中
# #     for row in df.itertuples(index=False, name=None):
# #         ws.append(row)
# #
# #     # 创建一个折线图
# #     chart = LineChart()
# #     chart.title = "Accuracy vs n"
# #     chart.x_axis.title = "n"
# #     chart.y_axis.title = "Accuracy"
# #
# #     # 添加数据到图表中
# #     data = Reference(ws, min_col=2, min_row=1, max_col=2, max_row=len(df))
# #     categories = Reference(ws, min_col=1, min_row=1, max_row=len(df))
# #     chart.add_data(data, titles_from_data=True)
# #     chart.set_categories(categories)
# #
# #     # 将图表添加到工作簿中
# #     ws.add_chart(chart, "E5")  # 可以根据需要调整位置
# #
# #     # 保存带有图表的工作簿
# #     chart_excel_path = f'{type}_{t}_chart.xlsx'
# #     wb.save(chart_excel_path)
# #
# #     print(f"Excel文件已保存到 {chart_excel_path}")
# #
# # create_excel(acc_dic)
