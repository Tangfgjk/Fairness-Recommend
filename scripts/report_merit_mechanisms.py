"""Summarize fixed-checkpoint probes; no training, no test-set parameter search."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/论文格式文档统计/merit_frequency_20261022/机制诊断_20261023'
RUN=ROOT/'runs/merit_frequency_20261022'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    lines=['# 第一轮去偏机制诊断（2026-10-23）','',
           '范围：三个数据集，seed2024；读取已完成 E0/E1 检查点及分数，新增固定强度 E2-logit 评价。未执行任何正式训练或参数更新。',
           '原结果保留，新输出仅位于本分析目录。正式 NOV 为 KG4Ex-NOV-Set。','',
           '## 主要发现','',
           '1. E1 相对 E0 的评分变化主要是所有学生共享的题目优先级变化；E1 的全题库共同题目效应占比也更高。这支持“换了一批共同推荐题”的机制解释，但描述性分解不是因果归因，也不是 Top-N 方差归因。',
           '2. 在已训练 E0 的 epoch25 检查点上，假设加入 E1 的 λ=0.1 公平项，其梯度可以大于 BCE；在实际 E1 的 epoch25 检查点上，当前样本的公平梯度已经很小。不能据此声称实际 E1 训练初期或全程都由公平项主导。',
           '3. E3 中大量题对仍满足 E1 条件；E3 正确解释为“不要求匹配”，不是“全部不匹配”。',
           '4. E2-logit 强度0.1在全部30个数据集×N通过质量预算，但通常比原概率校正0.1更弱。系数相同不代表干预强度相同，本次没有搜索测试集最优系数。','',
           '## 评分变化分解','',
           '先从 E1−E0 的 logit 矩阵去除学生统一加分，再分解为共同题目效应与学生×题目交互残差。','',
           '|数据集|共同题目效应占评分变化平方和比例|共同效应占比 E0→E1|题目平均分变化与rec频率 Spearman|',
           '|---|---:|---:|---:|']
    datasets=['Eedi','algebra2005','XES3G5M-sub-small']; metrics=[]
    for d in datasets:
        s=read(OUT/d/'score_decomposition.json')
        lines.append(f"|{d}|{s['score_change']['decomposition']['item_fraction_after_removing_user_offsets']:.2%}|{s['E0']['decomposition']['item_fraction_after_removing_user_offsets']:.2%} → {s['E1']['decomposition']['item_fraction_after_removing_user_offsets']:.2%}|{s['score_change']['mean_shift_rec_frequency_spearman']:.4f}|")
    lines+=['','显式 b[e] 变化无法单独解释这个现象：共同题目效应还会经过习题编码和评分网络形成。不能把该效应直接称为 bias 参数贡献率。','',
            '## 梯度探针','',
            '使用两个固定的1024样本混合关系训练批次，每次公平采样最多32学生、每学生4对，margin=0.1、λ=0.1。分别保存学生/习题/共享文本编码、ConvE、尾偏置及其他参数组的范数与夹角。每个探针恢复固定检查点和随机状态。没有优化器。','',
            '|数据集|在E0上假设加入公平项：范数比范围|E1末期公平项范数比范围|E0梯度方向余弦|','|---|---:|---:|---|']
    for d in datasets:
        g=read(OUT/d/'gradient_probes.json')
        get=lambda m:[r['groups']['ALL'] for r in g['results'] if r['method']==m and 'groups'in r]
        a,b=get('E0'),get('E1')
        ratios=lambda v:[r['weighted_fair_over_BCE'] for r in v]
        cosines=', '.join(format(r['cosine'],'.3f') for r in a)
        lines.append(f"|{d}|{min(ratios(a)):.2f}–{max(ratios(a)):.2f}|{min(ratios(b)):.4f}–{max(ratios(b)):.4f}|{cosines}|")
    lines+=['','末期公平梯度小，不证明此前没有发生强干预；某个样本池的 Hinge 已不激活，也不等于梯度实现失效。端点探针显示激活题对的两端梯度相反且非零；未激活题对梯度为零符合定义。固定两批不足以代表全部训练数据。','',
            '## E3 条件重叠','', '|数据集|E3中仍满足E1的题对比例|先学生内再学生间平均的比例|','|---|---:|---:|']
    for d in datasets:
        a=read(OUT/d/'E3_overlap_audit.json')
        lines.append(f"|{d}|{a['E3_matched_pair_fraction']:.2%}|{a['E3_student_macro_matched_fraction']:.2%}|")
    lines+=['','两种平均权重不同，须分别报告。现有 E3 无需作废，但其作用不能解释成严格的“移除全部匹配信息”。','',
            '## E2-logit 对照与逐N质量预算','',
            '输入为已保存的原始logit，不对饱和概率做反logit。频率特征沿用 minmax(log1p(train rec frequency))，系数固定0.1。原 E2 概率校正仍单列。正数为相对本轮 E0 提升。']
    for d in datasets:
        base=read(RUN/d/'E0/seed2024/eval/metrics.json')
        old=read(RUN/d/'E2/seed2024/eval/metrics.json')
        new=read(OUT/d/'E2_logit_alpha0p1/eval/metrics.json')
        lines += ['',f'### {d}', '',f"CFP：E0={base['CFP']['value']:.6f}；E2-prob={old['CFP']['value']:.6f}；E2-logit={new['CFP']['value']:.6f}。",
                  '', '|N|logit ΔAda|logit ΔNOV|预算通过|logit ΔDED|覆盖题数 E0→prob→logit|','|---|---:|---:|---|---:|---|']
        for n in range(5,51,5):
            a,b,c=[v['top_n'][str(n)] for v in (base,old,new)]
            da=c['Ada']-a['Ada'];dn=c['KG4Ex-NOV-Set']-a['KG4Ex-NOV-Set'];ok=da>=-.01-1e-12 and dn>=-.01-1e-12
            count=lambda v:round(v['ItemCoverage']*base['candidate_exercises'])
            lines.append(f"|{n}|{da:+.6f}|{dn:+.6f}|{'是' if ok else '否'}|{c['DED']-a['DED']:+.6f}|{count(a)} → {count(b)} → {count(c)}|")
            for method,v in [('E0',base),('E2_probability',old),('E2_logit_0.1',new)]:
                for key,value in v['top_n'][str(n)].items():
                    metrics.append([d,method,n,key,value,value-a[key]])
    lines+=['','## 基线核查及限制','',
            '归档 E0 与本轮重跑 E0 的已核查关键训练配置均一致，best epoch 均为25；Top-5 重合率仍只有约0.77–0.83。现有材料不足以恢复历史运行的全部批次轨迹和硬件数值行为。保留 Archived E0 与 Retrained E0 两层基线，当前方法比较仍以 Retrained E0 为准。','',
            '## 下一步建议','',
            '暂不加入 CompGCN，也不把 CFP 接近0.5作为最终成功目标。现有证据更支持需要处理共享题目优先级和实际曝光目标；干预强度的敏感性也值得用少量预登记实验确认，但不能把检查点探针当成训练全程证据。',
            '若继续路线A，应增加在线分项梯度监测，限定小规模配置，不再直接大规模扫 margin。若仍只能改善CFP，转向训练学生教育需求条件下的曝光优化，同时限制每学生教育适配损失。',
            '后续继续逐N固定 Ada、KG4Ex-NOV-Set 各下降≤0.01，完整报告 CFP、DED、曝光和列表差异。所有当前结论仍为单种子探索性结论。']
    (OUT/'机制诊断结论_20261023.md').write_text('\n'.join(lines),encoding='utf-8')
    with (OUT/'E2两种校正_完整指标与变化.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','method','N','metric','value','delta_vs_retrained_E0']);w.writerows(metrics)
    print('Report and complete comparison CSV saved.')


if __name__=='__main__':main()
