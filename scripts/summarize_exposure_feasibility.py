"""Summarize all preregistered offline allocations, including individual harms."""
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT,DATASETS,TOP_NS,read_json,sha256,write_json

OUT=ROOT/'runs/论文格式文档统计/merit_frequency_20261022/曝光可行性测试_20261023'
RUN=ROOT/'runs/merit_frequency_20261022'


def main():
    lines=['# 曝光可行性结论与下一步（2026-10-23）','',
           '本次没有训练或参数更新。固定K=100/200，使用测试人群的教育状态、正确历史和规则参照曝光做群体分配，是带评价信息的探索性可行性分析，不是训练模型性能，也不是最优解。',
           'F0为原E0；FR为规则参照；FQ只优化教育距离；FQE在同样候选和质量约束下额外优化真实曝光DED。所有Top-N来自同一排序前缀。',
           '已完成12组分配，全部120个数据集×候选范围×方法×N组合均满足总体平均 Ada、KG4Ex-NOV-Set 各下降不超过0.01。','',
           '## Top-10结果（两档全部呈现，不挑最好一档）','',
           '|数据集|K|方法|Ada|NOV|DED|覆盖题数|不同题集比例|','|---|---|---|---:|---:|---:|---:|---:|']
    harm=[];passed=0
    for d in DATASETS:
        base=read_json(RUN/d/'E0/seed2024/eval/metrics.json')
        a=base['top_n']['10']
        lines.append(f"|{d}|—|E0|{a['Ada']:.4f}|{a['KG4Ex-NOV-Set']:.4f}|{a['DED']:.4f}|{round(a['ItemCoverage']*base['candidate_exercises'])}|{a['UniqueSetRatio']:.4f}|")
        for k in [100,200]:
            for m in ['FQ','FQE']:
                p=OUT/d/f'K{k}_{m}'
                result=read_json(p/'eval/metrics.json')
                marker=read_json(p/'eval/evaluation_complete.json')
                if sha256(p/'eval/metrics.json')!=marker['metrics_sha256']:raise ValueError('Changed result')
                a=result['top_n']['10']
                lines.append(f"|{d}|{k}|{m}|{a['Ada']:.4f}|{a['KG4Ex-NOV-Set']:.4f}|{a['DED']:.4f}|{round(a['ItemCoverage']*result['candidate_exercises'])}|{a['UniqueSetRatio']:.4f}|")
                audit=read_json(p/'allocation_audit.json')['per_user_quality_changes']
                for j,n in enumerate(TOP_NS):
                    a=result['top_n'][str(n)];b=base['top_n'][str(n)]
                    assert a['Ada']>=b['Ada']-.01-1e-10 and a['KG4Ex-NOV-Set']>=b['KG4Ex-NOV-Set']-.01-1e-10
                    passed+=1
                    for name,key in [('Ada','delta_Ada'),('NOV','delta_NOV')]:
                        values=[r[key][j] for r in audit]
                        harm.append([d,k,m,n,name,len(values),sum(v<-.01-1e-10 for v in values),min(values)])
    lines+=['','## 怎么解释','',
            '1. 现有数据和候选空间内，确实能找到满足总体质量预算且明显缓解曝光集中与同质化的分配；此前E1失败不能简单归结为题库没有替代题。',
            '2. FQ本身已经显著提升覆盖和列表差异。FQE相较FQ进一步降低DED，但覆盖率不一定更高；DED追求接近规则需求，不等于最大覆盖或均等曝光。公平目标的独立贡献要相对于FQ解释，不能把全部收益归因于公平项。',
            '3. 这些结果使用了原ConvE查询中没有直接输入的细粒度认知状态和历史集合。离线分配可行不等于当前一维theta评分器能够学习它。下一步必须明确教育候选/状态在训练和推断时如何进入推荐流程，检查是否需要受控的认知接口增强。',
            '4. 总体质量保护不等于每学生保护；例如K=100 FQE在Top-10，Eedi有98/234位学生NOV下降超过0.01，Algebra2005为40/144，XES为102/241。应报告个体损失分布，并在下一版单独研究学生级适配约束，不把新增约束混称为当前总体预算。',
            '5. 测试人群规则曝光被用于目标分配，DED改善部分是直接优化它的结果；不能据此宣称真实学习机会公平，也不能用这些测试结果反复挑正式训练参数。',
            '6. 候选支持度文件给出每名学生的候选数量与集合差异。训练侧K=100/200分别在原rec50之外增加50/150个候选是规则嵌套的定义结果，不能独自证明质量可行；真正的证据来自实际分配和独立评价。','',
            '## 下一步','',
            '进入MAED的可执行设计：以训练学生合法历史构造按学生归一化的需求分布；明确训练/推断一致的教育候选接口；先离线检查可微曝光代理与真实Top-N的一致性。',
            '训练实验至少区分原ConvE、仅教育约束、教育约束+曝光目标；保留E1-v1和E2作对照。这样才能判断收益来自教育信息接入还是公平曝光目标。暂不直接加入CompGCN。',
            '继续使用全部N的总体Ada/NOV预算0.01，预先登记主曝光指标与辅助指标；学生级保护作为明确的附加设计单列。']
    assert passed==120
    (OUT/'可行性结论与下一步_20261023.md').write_text('\n'.join(lines),encoding='utf-8')
    with (OUT/'学生级质量损失审计.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','K','method','N','metric','students','students_drop_over_0.01','worst_student_delta']);w.writerows(harm)
    write_json(OUT/'verification_summary.json',{'allocations':12,'quality_checked_positions':passed,
          'quality_passed':passed,'training_updates':0,'individual_quality_not_constrained':True})
    print('All 120 quality checks passed; summary and individual-harm audit saved.')


if __name__=='__main__':main()
