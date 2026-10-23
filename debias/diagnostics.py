"""Read-only training-sampler and matching diagnostics; no optimization."""
import numpy as np
from .io import model_modules, write_json
from .prepare import baseline_folder
from .io import read_json


def sampling_counts(ctx, output):
    original = model_modules()
    from feature_loader import read_dict
    entities = read_dict(ctx.graph / "entities.dict")
    relations = read_dict(ctx.graph / "relations.dict")
    triples = original.read_triples(ctx.graph / "triples.txt", entities, relations)
    id2rel = {v:k for k,v in relations.items()}
    exercise_ids = sorted(v for k,v in entities.items() if k.startswith("ex"))
    entity_to_ex = {v:int(k[2:]) for k,v in entities.items() if k.startswith("ex")}
    positive = original.build_positive_tails_by_hr(triples)
    result = {"interpretation": "Actual deterministic per-epoch sampling counts, not gradient contributions; repeats every epoch", "seeds": {}}
    for seed in (2024,2025,2026):
        cfg = read_json(baseline_folder(ctx.name,seed) / "config.json")
        dataset = original.TripleDataset(triples,id2rel,exercise_ids,positive,cfg["negative_ratio"],seed)
        negatives = np.zeros(len(ctx.q),dtype=int)
        for sample_idx,(triple_idx,offset) in enumerate(dataset.sample_index):
            if offset:
                h,r,_ = dataset.triples[triple_idx]
                tail = dataset.sample_negative_tail(h,r,sample_idx)
                negatives[entity_to_ex[tail]] += 1
        result["seeds"][str(seed)] = {"positive_counts_per_epoch":ctx.frequency.tolist(),
                                     "negative_counts_per_epoch":negatives.tolist(),
                                     "positive_negative_ratio":[float(p/n) if n else None for p,n in zip(ctx.frequency,negatives)],
                                     "epochs":cfg["epochs"]}
    write_json(output,result)


def prepare_summary(folder, dataset):
    audit = read_json(folder / "matching_audit.json")
    coverage = read_json(folder / "high_exposure_coverage.json")
    train, test = audit["audit"]["E1"], audit["audit"]["evaluation"]
    lines = [f"# {dataset}：基线与匹配诊断", "", "此报告只使用既有分数与数据，没有训练新模型。", "",
             f"训练匹配：{train['users']} 位学生，{train['pairs']} 个题对；测试 CFP：{test['users']} 位学生，{test['pairs']} 个题对。", "",
             "E3 按每名学生的 log(1+频率) 差分层保持与 E1 相同的各层题对数。难度、区分度与 KC 内容未强制匹配，残余差异见 matching_audit.json。", "",
             "|种子|原 Top-10 曝光落在训练高频侧匹配题的比例|", "|---|---:|"]
    for seed, report in coverage.items():
        lines.append(f"|{seed}|{report['exposure_mass_on_high_side_item']:.4%}|")
    lines += ["", "该比例表示题目至少在一个训练学生的匹配池中出现，不代表所有收到该题的测试学生都被有效约束。", "",
              "|seed2024 高曝光题|Top-10 曝光人数|训练 rec 频率|高频侧题对数|涉及训练学生数|", "|---|---:|---:|---:|---:|"]
    for row in coverage['2024']['top20_exposed_exercises']:
        lines.append(f"|ex{row['exercise']}|{row['exposure']}|{row['rec_frequency']}|{row['high_side_pairs']}|{row['high_side_train_users']}|")
    lines += ["", "正式 NOV 为 KG4Ex-NOV-Set。基线 Ada 已与归档指标核对；Ep 使用同一份哈希验证后的原评分对应归档结果。新实验将重新模拟 Ep@0.5。",
              "CFP 仅覆盖有匹配题对的测试学生，其余教育和曝光指标覆盖完整测试人群。"]
    (folder / "诊断摘要_20261022.md").write_text("\n".join(lines),encoding="utf-8")
