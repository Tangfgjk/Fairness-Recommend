"""Document the no-training audit and freeze checkpoint identities, without moving models."""
import csv
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from debias.io import DATASETS,TOP_NS,read_json,write_json,sha256
from maed.io import lock_json,source_hash


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    out=ROOT/'runs/论文格式文档统计/maed_seed2024_mechanism_R_verified_20261024'
    run=ROOT/'runs/maed_controlled_monitored_seed2024_20261023'
    audit=read_json(out/'audit_summary.json');manifest=read_json(out/'audit_manifest.json')
    assert manifest['production_source_hash']==source_hash()==read_json(run/'protocol.json')['source_hash']
    shared=rows(out/'shared_item_counterfactual.csv');nov=rows(out/'NOV_loss_summary.csv')
    kc=rows(out/'algebra2005_KC_contribution.csv');grad=rows(out/'training_gradient_audit.csv')
    boundary=rows(out/'training_boundary_audit.csv');qrows=rows(out/'training_Q_components.csv')
    for reference in ('E0','M3'):
        for n in TOP_NS:
            total=sum(float(r['population_mean_NOV_contribution']) for r in kc if r['reference']==reference and int(r['N'])==n)
            mean=next(float(r['mean_delta_NOV']) for r in nov if r['dataset']=='algebra2005' and r['reference']==reference and int(r['N'])==n)
            assert abs(total-mean)<1e-12
    for d,a in audit.items():
        for row in a['personalization']:
            if row['method'] in a['Top10']:
                assert abs(row['mean_list_Jaccard_distance']-a['Top10'][row['method']]['ListJaccardDistance'])<1e-12
    # Candidate manifest fixes artifact identities; no checkpoint copy or optimizer update.
    candidate={'candidate':'MAED-v1-seed2024','training_source_hash':source_hash(),
        'observed_repository_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'head_scope':'observed repository HEAD during audit, not independent proof of training-time Git HEAD',
        'protocol_sha256':sha256(run/'protocol.json'),'beta':.5,'lambda_E':.01,'seed':2024,
        'selection':'minimum mean supervised BCE; last is diagnostic; no test-based reselection',
        'models':{},'datasets':{d:a['test_students'] for d,a in audit.items()}}
    for d in DATASETS:
        for method in ('M3','M4'):
            folder=run/d/method/'seed2024';done=read_json(folder/'training_complete.json')
            artifacts={p:sha256(folder/p) for p in ('best.pt','last.pt','best/uid_ex_scores.json','last/uid_ex_scores.json','run_config.json','initialization_audit.json')}
            assert artifacts['best.pt']==done['checkpoints']['best.pt'] and artifacts['last.pt']==done['checkpoints']['last.pt']
            candidate['models'][f'{d}/{method}']={'path':str(folder),'best_epoch':done['best_epoch'],'artifacts':artifacts}
    lock_json(out/'MAED_v1_candidate_manifest_seed2024_20261024.json',candidate)
    lines=['# MAED首轮机制审计与R规则对照（2026-10-24）','','## 1. 范围与完成状态','',
        '本次只读取seed2024的已保存模型结果、训练日志、正式数据和历史R结果，不训练、不扩展种子、不改模型或超参数。原正式训练目录及原论文统计目录保持不变。新增审计脚本位于scripts，未改变生产训练源码摘要及原运行合同。',
        f"共核对{manifest['source_count']}份输入文件来源哈希。R与E0/M3/M4的数据指纹、测试学生、匹配题对和NOV/Ep口径一致；已从分数重算排序教育/曝光指标，Ep复核保存的逐学生模拟结果，不重新模拟。",
        '已生成三个带R的论文格式HTML，每指标一张表。R为未训练规则参照，不参与模型最优加粗、次优下划线；其全部数值保留。',
        '候选模型身份由MAED_v1_candidate_manifest_seed2024_20261024.json锁定：六个模型的best/last权重、分数、配置和初始化哈希，beta=0.5、lambdaE=0.01、seed2024、原25轮和最低监督BCE选择策略。',
        '', '## 2. R规则参照揭示的取舍', '',
        'R直接按正式教育距离从小到大推荐，未训练。R的DED=0、RuleOverlap=1由定义决定，不能据此宣称真实教育公平最好。但R的Ada、NOV、Ep、覆盖和集中度仍有实际对照价值，不能因为不是训练模型而忽略。',
        '', '|数据集|方法|Ada@10|NOV@10|Ep@0.5@10|ItemCoverage@10|','|---|---|---:|---:|---:|---:|']
    for d,a in audit.items():
        for method in ('E0','M3','M4','R'):
            value=a['Top10'][method]
            lines.append('|'+d+'|'+method+'|'+ '|'.join(f'{value[k]:.6f}' for k in ('Ada','KG4Ex-NOV-Set','Ep_t0.5','ItemCoverage'))+'|')
    lines+=['',
        '重要发现：R在三个数据集全部30个数据集×N位置上的Ada均高于M4；在习题/KC覆盖及Gini上也优于M4。因此现有结果支持M4优于原学习模型E0/M3，不能写成全面优于直接教育规则。',
        'M4的模拟Ep高于R：Eedi 9/10档、XES 10/10档、Algebra2005 5/10档；Algebra2005的NOV有6/10档高于R。模拟Ep优势不是实际学习效果的证明。',
        '', '## 3. 共同习题分数变化与个性化残差', '',
        '取同一测试群体的M4减M3 logit变化，先减去每位学生的行均值（不影响排序），再拆成所有学生共享的逐题均值变化和学生残差。用平方能量占比描述；该数值不是公平损失的因果贡献率。另做每个模型逐学生z-score后的敏感性分析。',
        '', '|数据集|共同项能量占比|逐学生标准化后的共同项占比|M3 DED@10|仅加共同项 DED@10|M4 DED@10|','|---|---:|---:|---:|---:|---:|']
    for d,a in audit.items():
        decomp=a['score_decomposition']
        cf=next(r for r in shared if r['dataset']==d and r['N']=='10' and r['metric']=='DED')
        lines.append(f"|{d}|{decomp['M3_to_M4']['shared_item_energy_fraction']:.2%}|{decomp['row_standardized_sensitivity']['shared_item_energy_fraction']:.2%}|{float(cf['M3']):.6f}|{float(cf['shared_item_only']):.6f}|{float(cf['M4']):.6f}|")
    lines+=['',
        '仅加共同项是事后从同一测试群体分解得到的诊断性重排，不是额外训练的模型，也不是可部署且已验证的独立方法；不参与模型表格最优排名，不计算新的模拟Ep。',
        'Eedi的共同项已能改善曝光，但完整M4进一步降低DED并提高Ada；XES仅共同项使DED变差，完整M4的学生残差对最终排序很重要。Algebra2005的共同项已能重现多数DED改善，不能声称其收益主要来自新增个性化。',
        '', '|数据集|M3规则列表差异相关@10|M4规则列表差异相关@10|','|---|---:|---:|']
    for d,a in audit.items():
        pair={r['method']:r for r in a['personalization']}
        lines.append(f"|{d}|{pair['M3']['rule_list_distance_model_list_distance_spearman']:.6f}|{pair['M4']['rule_list_distance_model_list_distance_spearman']:.6f}|")
    lines+=['',
        '这里计算学生两两规则推荐集合距离与模型推荐集合距离的Spearman相关，是依赖学生对的描述性关联，不报告独立样本p值。与掌握向量距离的相关性没有一致提高，XES从约0.264降到0.241，不能把规则列表对应改善概括成所有认知维度个性化全面提高。',
        '', '## 4. Algebra2005的NOV下降：学生与KC明细', '',
        '将推荐KC集合记为K、历史答对KC集合记为H。NOV=1−|K∩H|/|K∪H|。逐学生对比M4与E0、M3，记录新增/移除习题和KC、KC是否属于H、NOV与Ada变化、末5题首次带入的KC。',
        '将NOV变化精确、对称地分为交集变化项与并集变化项；所有KC的记账贡献求和等于实际NOV变化。这是集合变化的数学记账，不是训练过程的因果解释。',
        '', '|N|平均NOV变化 vs E0|严格损失>0.03人数（精确）|恰为0.03人数|旧浮点>0.03人数|','|---:|---:|---:|---:|---:|']
    for row in audit['algebra2005']['novaudit']:
        lines.append(f"|{row['N']}|{row['mean_delta_NOV']:+.6f}|{row['loss_gt_0p03_students']}/144|{row['loss_eq_0p03_students']}|{row['legacy_float_loss_gt_0p03_students']}|")
    fifty=next(r for r in audit['algebra2005']['novaudit'] if r['N']==50)
    lines+=['',
        f"Top50总NOV变化{fifty['mean_delta_NOV']:+.6f}，其中与历史答对KC交集变化项{fifty['known_overlap_contribution']:+.6f}，未答对过KC引起的并集变化项{fifty['unseen_union_contribution']:+.6f}。主要负向项来自历史答对KC重合增加，未见该负向项被新KC扩展充分抵消。",
        '负向记账较大的新增历史答对KC包括kc74、kc89，分别涉及87、85位学生（可能重叠，不可相加当作人数）。KC按正式Q矩阵列编号，不臆造知识点名称。',
        f"严格NOV损失超过0.03的学生为{fifty['loss_gt_0p03_students']}/144（{fifty['loss_gt_0p03_rate']:.2%}），不是仅少数学生；最差10%学生平均损失{fifty['worst_10pct_mean_loss']:.6f}，最大损失{fifty['max_loss']:.6f}。",
        '数值边界修正：学生uid436、uid568的集合计数给出精确变化−3/100。旧浮点值为−0.030000000000000027，导致原表计为84人；新审计按整数交叉相乘将其列为恰好0.03，严格超出为82人。原表和所有原始指标不改写；新CSV同时保留精确人数、边界人数及旧浮点人数。该修正不改变任何模型分数或群体平均值。',
        '按你的最新取舍，Algebra2005的NOV下降作为可接受的Ada/曝光权衡继续完整报告，不再作为本轮候选模型的阻断项；原0.03参考门槛的历史验收仍保留。历史答对KC的再次推荐也可能具有遗忘复习价值，不能把集合NOV下降直接等同教育推荐变差。',
        '', '## 5. 训练机制审计', '',
        '|数据集|末轮Top10软质量占比|加权曝光/BCE梯度范数比中位数|最大比值|梯度余弦为负比例|','|---|---:|---:|---:|---:|']
    for d in DATASETS:
        end=next(r for r in boundary if r['dataset']==d and r['method']=='M4' and r['epoch']=='25' and r['N']=='10')
        gs=[r for r in grad if r['dataset']==d]
        ratio=np.array([float(r['weighted_exposure_over_BCE']) for r in gs]);cos=np.array([float(r['BCE_exposure_gradient_cosine']) for r in gs])
        lines.append(f"|{d}|{float(end['actual_TopN_soft_mass_fraction']):.2%}|{np.median(ratio):.4f}|{ratio.max():.4f}|{np.mean(cos<0):.2%}|")
    lines+=['',
        '每个数据集M4记录25轮×3个有效批次=75个梯度点。这是稀疏监测，不代表每批梯度分布；范数比偶尔超过1或夹角冲突不自动证明训练错误。',
        '固定温度确有漂移：Eedi和XES末轮全部N均存在机制标记，软/硬对应不如早期，但实际硬DED仍改善。因此保留v1结果和冻结温度，不将代理称为完全稳定，也不根据这轮测试结果自动重选温度。',
        'Q的全局变化斜率没有显示接近−1的整体完全抵消，且审计学生的core/fused Top10仍明显不同；这不能排除局部抵消。core与训练rec频率仍高相关，不能宣称已完全去除频率信息。',
        '', '## 6. CFP与研究主张', '', '|数据集|CFP E0|CFP M4|匹配学生/全部测试学生|','|---|---:|---:|---:|']
    for d,a in audit.items():
        lines.append(f"|{d}|{a['CFP']['E0']['value']:.6f}|{a['CFP']['M4']['value']:.6f}|{a['CFP']['M4']['users']}/{a['test_students']}|")
    lines+=['',
        'CFP没有稳定、充分降低，XES甚至略高于E0。证据支持“推荐曝光更接近教育需求规则、曝光集中度下降”，不支持“教育匹配条件下高频评分优势已普遍消除”。',
        'DED与规则列表相似性依赖同一教育规则，不是两项独立公平证明；Gini、Coverage和学生列表距离提供补充证据，但仍非现实教育机会公平或真实学习增益的证明。',
        '', '## 7. 文件与复现', '',
        '最终输出目录：`runs/论文格式文档统计/maed_seed2024_mechanism_R_verified_20261024/`。',
        '', '|文件|内容|','|---|---|',
        '|三个`*_one_metric_per_table_with_R.html`|加入R参照后的完整论文表格|',
        '|MAED_v1_candidate_manifest_seed2024_20261024.json|候选配置和模型/分数身份|',
        '|training_boundary_audit.csv、training_soft_hard_transitions.csv|逐轮逐N温度和软/硬曝光|',
        '|training_gradient_audit.csv、training_Q_components.csv|梯度比例/夹角与Q作用|',
        '|shared_item_counterfactual.csv、personalization_pair_associations.csv|共同项诊断与学生差异关联|',
        '|student_NOV_KC_changes.csv、algebra2005_KC_contribution.csv|学生/习题/KC增减与精确记账|',
        '|NOV_loss_summary.csv、M4_prefix_new_KCs.csv|全N尾部风险、精确边界和列表后缀KC|',
        '|M4_vs_R_all_N.csv、all_metrics_with_R.csv|逐N完整R对照与论文源数据|',
        '|audit_manifest.json、audit_protocol.json|数据/源码来源及输出约定|',
        '', 'PowerShell复现（不训练）：', '', '```powershell',
        "$repo = 'C:\\Users\\21709\\Desktop\\Fairness-Recommend\\Fairness-Recommend-New'".replace('\\\\','\\'),
        "$py = 'C:\\Users\\21709\\.conda\\envs\\rtx5060\\python.exe'".replace('\\\\','\\'),
        'Set-Location -LiteralPath $repo',
        r'& $py -u .\scripts\audit_maed_seed2024.py',
        "if ($LASTEXITCODE -ne 0) { throw '机制审计失败' }",
        r'& $py -u .\scripts\summarize_maed_seed2024_audit.py',
        "if ($LASTEXITCODE -ne 0) { throw '审计总结或候选身份校验失败' }",'```',
        '', '本次不开展多随机种子，不加入CompGCN/KPPD/Post，不修改数学目标。当前可继续保留MAED-v1作为有价值候选，解释需同时包含R强参照、共享分数变化、局部频率偏好残留和学生级风险。']
    text='\n'.join(lines)+'\n'
    (out/'机制审计与R对照结论_20261024.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/MAED首轮机制审计与R规则对照_20261024.md').write_text(text,encoding='utf-8')
    print(out/'机制审计与R对照结论_20261024.md')


if __name__=='__main__':main()
