"""Read-only reanalysis of seed2024; no training or parameter selection."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/merit_frequency_20261022'
OUT=ROOT/'runs/论文格式文档统计/merit_frequency_20261022/质量约束复评_20261023'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lines=['# Ada/NOV 下降预算 0.01：逐 N 复评（2026-10-23）','',
           '范围：三个数据集、seed2024；相对于同轮重跑 E0。正式 NOV 为 KG4Ex-NOV-Set。',
           '判定：ΔAda≥−0.01 且 ΔNOV≥−0.01；提升不受限。以未四舍五入值判定。',
           '本预算是在看到本轮结果后确定，属于探索性复评；不能写成事前登记的验证结论。',
           '沿用原方案的核心诊断 CFP 和 DED，同时报告二者；Coverage/Gini 是辅助证据。CFP 接近0.5仅表示匹配池内较少单向频率优势，过低也可能是反向偏好；DED 更低仅表示更接近规则参照。',
           'CFP不随N变化，也仅覆盖匹配测试子人群，不能把它当成每个N的公平胜率。','']
    rows=[]
    for dataset in ['Eedi','algebra2005','XES3G5M-sub-small']:
        results={}
        for m in ['E0','E1','E2','E3','E4']:
            path=RUN/dataset/m/'seed2024/eval/metrics.json'
            marker=json.loads((path.parent/'evaluation_complete.json').read_text(encoding='utf-8'))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=marker['metrics_sha256']:
                raise ValueError(f'Changed metrics: {path}')
            results[m]=json.loads(path.read_text(encoding='utf-8'))
        lines += [f'## {dataset}', '', '各格为 ΔAda / ΔNOV；✗ 表示至少一个超预算。正数为提升。','',
                  '|N|E1|E2|E3|E4|','|---|---|---|---|---|']
        passed={m:[] for m in ['E1','E2','E3','E4']}
        ded={m:[] for m in passed}
        for n in range(5,51,5):
            b=results['E0']['top_n'][str(n)]
            cells=[]
            for m in passed:
                a=results[m]['top_n'][str(n)]
                da=a['Ada']-b['Ada'];dn=a['KG4Ex-NOV-Set']-b['KG4Ex-NOV-Set']
                ok=da>=-.01-1e-12 and dn>=-.01-1e-12
                if ok:passed[m].append(n)
                if a['DED']<b['DED']-1e-12:ded[m].append(n)
                cells.append(f'{da:+.4f} / {dn:+.4f}'+('' if ok else ' ✗'))
                rows.append([dataset,m,n,da,dn,ok,a['DED']-b['DED'],
                             a['ItemCoverage']-b['ItemCoverage'],a['KCCoverage']-b['KCCoverage']])
            lines.append('|'+str(n)+'|'+'|'.join(cells)+'|')
        lines+=['','|方法|同时通过的N|CFP：E0→方法|DED改善的N|','|---|---|---|---|']
        for m in passed:
            lines.append(f"|{m}|{','.join(map(str,passed[m])) or '无'}|{results['E0']['CFP']['value']:.4f} → {results[m]['CFP']['value']:.4f}|{','.join(map(str,ded[m])) or '无'}|")
        lines+=['']
    lines+=['## 解读','',
            'E2在全部30个数据集×N组合通过质量预算，是当前质量代价最小的去偏对照；CFP均小幅改善，DED在Eedi改善9/10个N、XES改善8/10个N，Algebra2005仅改善1/10个N。因此可称保守候选，不能称三个数据集全面公平改善。',
            'E1通过7/30，E3通过1/30，E4通过5/30。E1的主要问题在Eedi是NOV下降，在Algebra2005是短列表Ada下降，在XES是全部N的Ada下降；E1的DED在三个数据集全部N均变差。',
            '不能只报告通过的N，也不能把辅助Coverage改善替代原定核心指标。当前单种子结果支持保留E2对照、继续改进E1的损失强度/约束形式，不支持宣布E1为最终模型。']
    (OUT/'逐N复评结论.md').write_text('\n'.join(lines),encoding='utf-8')
    with (OUT/'逐N指标变化与质量门槛.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','method','N','delta_Ada','delta_KG4Ex_NOV_Set','quality_pass','delta_DED','delta_ItemCoverage','delta_KCCoverage']);w.writerows(rows)
    print(OUT)


if __name__=='__main__':main()
