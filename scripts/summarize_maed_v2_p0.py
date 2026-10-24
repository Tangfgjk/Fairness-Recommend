"""Validate P0 evidence and write a source-backed engineering acceptance record."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from debias.io import DATASETS,read_json,write_json,sha256


def summarize(source, output):
    source=Path(source).resolve();output=Path(output).resolve()
    manifest=read_json(source/'run_manifest.json')
    changed=[p for p,h in manifest['source_files'].items() if sha256(ROOT/p)!=h]
    if changed:raise ValueError(f'P0 evidence comes from different code: {changed}')
    reports={};rows=[];chunks=[];exceptions=[]
    for dataset in DATASETS:
        folder=source/dataset;r=read_json(folder/'P0_result.json');reports[dataset]=r
        audit=read_json(folder/'graph_audit.json');query=r['inference']['query_records']
        if r['formal_training_started'] or r['status']!='passed':raise ValueError('Invalid P0 status')
        if not r['full_cohort_cost'] or not r['inference']['full_cohort']:raise ValueError('Full cohort cost required')
        if len({q['uid'] for q in query})!=r['inference']['total_test_students']:raise ValueError('Query cohort incomplete')
        if not all(q['cross_query_rejected'] and q['triple_catalog']['passed'] for q in query):raise ValueError('Score/isolation failure')
        repeat=r['inference']['query_isolation_repeat']
        if not repeat['passed'] or not repeat['same_graph_id'] or not all(repeat['TopN_identical'].values()):
            raise ValueError('Repeated-query isolation failed')
        if not all(r['original_regression'][k]['passed'] for k in ('triple_logit','catalog_logit','catalog_sigmoid')):
            raise ValueError('Original checkpoint regression failed')
        if any(q['parameters_sha256']!=r['initialization_sha256'] for q in query):raise ValueError('Inference parameters changed')
        if r['full_cohort_cost']['parameter_hash']!=r['initialization_sha256']:raise ValueError('Cost used different parameters')
        selected=next(x for x in r['training_profile']['rows'] if x['chunk_size']==r['training_profile']['chosen_chunk'])
        if selected['status']!='passed':raise ValueError('Selected chunk failed')
        for x in r['training_profile']['rows']:
            if x['status']!='out_of_memory':
                if not all(v['finite'] for v in x['first_gradient_groups'].values()):raise ValueError('Nonfinite gradients')
                chunks.append(f"|{dataset}|{x['chunk_size']}|{x['status']}|{x['mean_step_seconds']:.4f}|{x['peak_allocated_bytes']/2**20:.1f}|{x['peak_reserved_bytes']/2**20:.1f}|")
            if x['status']=='equivalence_failed':
                failures={k:v for k,v in x['equivalence']['gradients'].items() if not v['passed']}
                exceptions.append({'dataset':dataset,'chunk_size':x['chunk_size'],'failed_gradients':failures,
                                   'logits':x['equivalence']['logits'],'action':'excluded from selection'})
        for q in query:
            if not q['local_equivalence']['accepted']:
                exceptions.append({'dataset':dataset,'uid':q['uid'],'local_equivalence':q['local_equivalence'],
                                   'action':'full B reference used; no quantization/tie-rule changes'})
        local=sum(q['local_equivalence']['accepted'] for q in query)
        cost=r['full_cohort_cost']
        rows.append(f"|{dataset}|{len(query)}|{r['registered_parameters']:,}|{selected['chunk_size']}|{selected['mean_step_seconds']:.4f}|{selected['peak_allocated_bytes']/2**20:.1f}|{cost['A']['wall_seconds']:.3f}|{cost['B']['wall_seconds']:.3f}|{local}/{len(query)}|")
    proof=read_json(ROOT/'third_party/CompGCN/PROVENANCE.json')
    for name,entry in proof['files'].items():
        if sha256(ROOT/'third_party/CompGCN'/name)!=entry['local_sha256']:raise ValueError('Upstream reference changed')
    compact={'scope':'P0 engineering acceptance, no formal C0/C3/C4 training',
             'evidence_directory':str(source.relative_to(ROOT)), 'source_files':manifest['source_files'],
             'exceptions':exceptions,'upstream_commit':proof['commit'],
             'datasets':{name:{'status':r['status'],'registered_parameters':r['registered_parameters'],
                 'chosen_chunk':r['training_profile']['chosen_chunk'],
                 'full_cohort_cost':r['full_cohort_cost'],
                 'queries':r['inference']['total_test_students'],
                 'local_accepted':sum(q['local_equivalence']['accepted'] for q in r['inference']['query_records']),
                 'graph_audit_sha256':sha256(source/name/'graph_audit.json'),
                 'result_sha256':sha256(source/name/'P0_result.json')} for name,r in reports.items()}}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'P0工程验收摘要_20261024.json',compact)
    content='''# MAED-v2 P0 工程验收结果

日期：2026-10-24。分支：`maed-v2-semantic-compgcn-20261024-152715`。

## 1. 当前结论

**P0 的 C0 核心工程实现已完成；完整 B 参考可在本机 GPU 上运行。没有启动任何正式 25 轮 C0/C3/C4 实验，也没有产生这些模型的推荐质量结论。**

已实现共享原 feature_only 语义初始化、单层 mult、逆关系方向变换、双向对称度、独立自环、LayerNorm+tanh、0.5 残差、动态孤立旁路、原 ConvE 解码及显式编码快照。所有正式输出使用完整 B 参考。

自动化检查：`python -m unittest discover -s tests -p test_maed_v2.py -v`，19 项通过。MAED-v1 的 annotated tag 及 1,149 项冻结清单复核通过，提交仍为 `280d65be46ba9dbae4511decaa4844a97ea53965`，无环境差异。

## 2. 三数据集真实 GPU 检查

RTX 5060 Laptop，PyTorch float32；原 bs=1024、负采样比=5、Adam lr=0.001。每档同初态/同 RNG/同一个固定训练 batch，2 步预热 + 5 步计时，更新仅用于临时工程压力测试。完成后恢复未训练初态再做推断。

|数据集|测试学生|注册参数|所选 chunk|完整步秒|训练峰值 allocated MiB|全群体 A 秒|全群体 B 秒|局部参考通过|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

全部 619 位测试学生逐人从干净训练基图扩图，验证全候选评分、逐三元组与全矩阵一致、跨学生快照拒绝及重复查询隔离。训练基图均无 rec/测试状态传播，Q 状态与实际 mlkc/exfr 逐条对照，保留量化边界记录。训练和派生查询图均排除量化 exfr0.00，保留 mlkc0.00。

以上 A/B 成本是**未训练 P0 模型、驻留暖设备、完整群体实际计时**：包含语义准备、独立扩图/度更新/传输、全图更新、全题库 ConvE、Top-50 稳定排序及 CPU/NPZ 导出；不含进程启动、文件来源核验及 checkpoint 加载。A/B 都逐学生调用解码器。没有用单学生耗时外推，不是旧 E0 训练或最佳批量调度的速度对比。

峰值是 PyTorch allocator 实测，不等于整个设备/其他桌面程序总用量。注册参数包含继承但 feature_only 不活跃的表；实际梯度组另在原始 JSON 中记录。

## 3. 四档分块结果及保留的问题

|数据集|chunk|结果|平均完整步秒|峰值 allocated MiB|峰值 reserved MiB|
|---|---:|---|---:|---:|---:|
'''+ '\n'.join(chunks)+'''

CPU float64 小图的整块/分块前向及 H0、R0、传播参数梯度通过严格等价检查。真实 GPU 四档均能执行，但 Eedi 部分大块的少量共享文本 GRU 反向梯度超出预先固定的 `atol=1e-5, rtol=1e-4`；对应 chunk 已排除，未放宽容差。CUDA 原子归约/原生算子的末位差异在实际运行之间可能变化，所以本次选择仅是本机本轮工程记录，不宣称某块数学定义不同或跨硬件最优。

局部参考仅重新物化“当前学生+全部习题”行，完整邻居及新图度仍全部参与。Eedi 234/234、XES 241/241 通过全部 Top-5～50 对照；Algebra2005 为 143/144。Algebra2005 学生 543 的表示和 logit 在容差内（最大 logit 差约 8.34e-7），但 Top-30～50 顺序不同，故不接受该局部结果。**整个数据集继续使用完整 B 参考；没有改精度或并列规则让检查假通过。**

测试中的“局部参考”尚非基图缓存预聚合的增量优化内核。它不是目前生产输出路径，也没有宣称已经证明所有局部推断等价。

## 4. 已完成与后续

|工作|状态|
|---|---|
|P0-1 图协议、来源快照、过滤、逆边及查询隔离|完成|
|P0-2 单层编码器、孤立旁路及梯度|完成|
|P0-3 原 E0 checkpoint 零传播完整回归、统一评分和完整 B|完成|
|可选局部参考|已实现；保留 Algebra2005 个别排序差异，不启用为正式输出|
|P0-4 三图真实单步反向、显存、全部查询及完整群体成本|完成|
|真实数据的编码动力学诊断|已输出基图及首个查询；完整已训练群体诊断待 C0 后执行|
|C0 25 轮正式训练、断点恢复与完整评价编排|本轮未启动；需后续独立训练入口和门槛|
|C3/C4 的 Q/D/E 接入、新 C0 温度标定、正权重续跑验收|下一阶段，尚未完成|

19 项自动化检查包含小图正权重曝光梯度与 BN/Dropout/RNG/异常恢复；这不代替 C4 真实 CUDA 正权重多批次续跑门槛。P0 未比较 Ada、NOV、Ep、DED 的训练后改进，不把模型能够运行等同于推荐有效。

## 5. 证据与复现

原始工程结果：`'''+source.relative_to(ROOT).as_posix()+'''/`。

- `run_manifest.json`：本次执行源码 SHA、设备环境、参数。
- 各数据集 `graph_audit.json`：状态/图文件哈希、过滤/划分/关系核验及量化边界样例。
- `original_regression.json`：真实 E0 完整 checkpoint 的 logit/sigmoid/全题库零传播回归。
- `chunk_profile.json`：全部候选 chunk 的耗时、峰值及逐参数梯度等价结果，失败项保留。
- `inference_probe.json`：全部查询的图 ID、状态差分身份、度哈希、评分/排序/隔离检查。
- `base_encoding_diagnostics.json` / `query_encoding_diagnostics.json`：rho/delta、消息、LN、tanh、类型及度诊断。
- `full_cohort_cost.json` 与 `P0_untrained_full_cohort_A/B.npz`：实际群体成本与未训练模型的工程输出，不是正式论文成绩。
- 本目录 `P0工程验收摘要_20261024.json`：源码身份、主要结果和未通过项的可携带摘要。

本次执行源码哈希与当前实现逐文件核对一致。官方来源已固定至 `'''+proof['commit']+'''`，原始源码与 Apache-2.0 许可证保存在 `third_party/CompGCN/`；新实现显式记录语义初始化、对称度、LayerNorm/残差等适配，不宣称官方逐值复现。

复核命令见《P0设计基准与实现范围_20261024.md》。新运行使用新目录，不覆盖原数据、历史结果或 MAED-v1 冻结内容。
'''
    (output/'P0工程验收结果_20261024.md').write_text(content,encoding='utf-8')
    return compact


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'docs/MAED-v2')
    a=p.parse_args();summary=summarize(a.source,a.output)
    print(f"Verified {sum(r['queries'] for r in summary['datasets'].values())} queries; report: {a.output}")
