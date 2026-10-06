#!/usr/bin/env python3
"""Six reproducible paper figures from saved evaluations; no model calls."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
import tempfile
import os

# Avoid modifying the user's global matplotlib cache.
_cache = tempfile.TemporaryDirectory(prefix='paper-mpl-')
os.environ.setdefault('MPLCONFIGDIR', _cache.name)
os.environ.setdefault('XDG_CACHE_HOME', _cache.name)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'results/ifs-2027/analysis'
MODELS = ['deepseek-v4.1-flash:cloud', 'glm-5.3-flash:cloud', 'gpt-6-astra']
NAMES = ['DeepSeek', 'GLM', 'Astra']
COLORS = ['#eb6834', '#2a78d6', '#1baf7a']
MARKERS = ['o', 's', '^']
POLICIES = ['independent_max', 'one_to_one']
PCOLORS = ['#2a78d6', '#eb6834']
GUIDED = {'guided_baseline', 'guided_headline', 'guided_curve', 'astra_cross_corpus'}
WIDTH = 4.8


def read(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def yes(value):
    return str(value).lower() == 'true'


def unique(rows, key):
    result = {}
    for r in rows:
        k = key(r)
        if k in result:
            raise ValueError(f'Duplicate key: {k}')
        result[k] = r
    return result


def save(fig, out, name, data):
    # No bbox_inches=tight: preserve exact final physical dimensions.
    fig.savefig(out / (name + '.pdf'))
    fig.savefig(out / (name + '.png'), dpi=220)
    plt.close(fig)
    with (out / (name + '.csv')).open('w', newline='', encoding='utf-8') as f:
        fields = list(dict.fromkeys(k for row in data for k in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(data)


def score(scores, rid, policy, tau=.4):
    row = scores.get((rid, tau, policy))
    return float(row['f1_total']) if row else None


def guided(row):
    return row['arm'] in GUIDED and yes(row['metamodel_block'])


def policy_legend(fig, y=1):
    fig.legend([Line2D([], [], color=PCOLORS[i], linestyle=['-', '--'][i],
                       marker=['o', 's'][i], lw=2) for i in range(2)],
               ['Independent-max', 'One-to-one'], loc='upper center',
               bbox_to_anchor=(.5,y), ncol=2, frameon=False)


def inflation(wide, scores, out):
    data=[]; excluded=[]
    for rid, w in wide.items():
        a=scores.get((rid,.4,'independent_max')); b=scores.get((rid,.4,'one_to_one'))
        if not a or not b: continue
        if float(b['precision']) <= 0:
            excluded.append(rid); continue
        data.append(dict(run_id=rid, model_id=w['model_id'], corpus=w['corpus'],
            size_ratio=int(a['n_generated'])/int(a['n_reference']),
            precision_ratio=float(a['precision'])/float(b['precision'])))
    fig,ax=plt.subplots(figsize=(WIDTH*.75,2.7),layout='constrained')
    for model,name,color,marker in zip(MODELS,NAMES,COLORS,MARKERS):
        rs=[r for r in data if r['model_id']==model]
        ax.scatter([r['size_ratio'] for r in rs],[r['precision_ratio'] for r in rs],
                   s=13,c=color,marker=marker,alpha=.55,label=name,linewidths=.3)
    rho=float(spearmanr([r['size_ratio'] for r in data],[r['precision_ratio'] for r in data]).statistic)
    ax.set(xscale='log',xlabel='Generated / reference feature count',ylabel='Precision ratio\n(independent / one-to-one)')
    ax.axvline(1,color='.45',ls=':',lw=1)
    ax.set_xticks([.1,.25,.5,1,2,4,8],['0.1','0.25','0.5','1','2','4','8'])
    ax.minorticks_off()
    ax.text(1.03,.96,'reference size',transform=ax.get_xaxis_transform(),rotation=90,
            va='top',ha='left',fontsize=7,color='.4')
    ax.text(.03,.67,f'Spearman ρ = {rho:.3f}\nn = {len(data)}',transform=ax.transAxes,va='top')
    ax.legend(frameon=False,fontsize=7,loc='upper left')
    save(fig,out,'fig_inflation',data)
    return dict(n=len(data),spearman_rho=rho,excluded_zero_denominator=excluded,
                caveat='Pooled descriptive association across heterogeneous arms; metrics share size-related components, not a causal effect.')


def granularity(wide,scores,out):
    fig,axes=plt.subplots(1,4,figsize=(WIDTH,2.2),sharey=True,layout='constrained')
    data=[]; plotted=[]
    for ax,(corpus,model) in zip(axes.flat,[(c,m) for c in ['repair','federation'] for m in MODELS[:2]]):
        ns=[1,5,10,20,54] if corpus=='repair' else [1,5,10,23]
        labels=[]
        for n in ns:
            rs=[w for w in wide.values() if guided(w) and w['corpus']==corpus and w['model_id']==model
                and int(w['N'])==n and w['grounding']=='rag' and w['ordering_id']=='primary' and int(w['k_doc'])==5]
            completed=sum(w['inventory_status']=='completed' for w in rs)
            labels.append(f'{n}\n{completed}/{len(rs)}')
            if rs and completed==0:
                ax.text(ns.index(n),.05,'no\noutput',transform=ax.get_xaxis_transform(),
                        ha='center',va='bottom',fontsize=6)
            for policy in POLICIES:
                vals=[score(scores,w['run_id'],policy) for w in rs]
                vals=[v for v in vals if v is not None]
                mean=st.mean(vals) if vals else None
                data.append(dict(corpus=corpus,model_id=model,N=n,policy=policy,planned=len(rs),
                    completed=completed,scored=len(vals),mean_f1=mean))
        for p,policy in enumerate(POLICIES):
            rs=[r for r in data if r['corpus']==corpus and r['model_id']==model and r['policy']==policy]
            ys=[r['mean_f1'] if r['mean_f1'] is not None else np.nan for r in rs]
            plotted.extend(y for y in ys if np.isfinite(y))
            ax.plot(range(len(ns)),ys,color=PCOLORS[p],ls=['-','--'][p],marker=['o','s'][p],ms=4)
            if p==1 and any(np.isfinite(ys)):
                peak=max(y for y in ys if np.isfinite(y))
                for x,y in enumerate(ys):
                    if y==peak: ax.scatter(x,y,s=80,facecolors='none',edgecolors='black',zorder=5,lw=.8)
        ax.set_xticks(range(len(ns)),labels,fontsize=5.5)
        ax.set_title(f'{corpus.title()}\n{NAMES[MODELS.index(model)]}',fontsize=8)
    limits=(.2,.8) if plotted and min(plotted)>=.2 and max(plotted)<=.8 else (0,1)
    for ax in axes.flat: ax.set_ylim(*limits)
    axes[0].set_ylabel('Mean full-reference F1',fontsize=7)
    fig.supxlabel('N (completed/planned)',fontsize=7)
    policy_legend(fig,1.065)
    fig.set_constrained_layout_pads(h_pad=.02,w_pad=.015)
    # Reserve legend within the physical figure.
    fig.get_layout_engine().set(rect=(0,0,1,.84))
    for legend in fig.legends: legend.set_bbox_to_anchor((.5,1))
    save(fig,out,'fig_granularity',data)
    return dict(ylim=limits,note='Means conditional on scored completions; gaps are not zeros. Circled peak is descriptive, not estimated saturation.')


def closure(wide,scores,semantic,out):
    series_colors = {
        ('independent_max', 'weak'): '#2a78d6',
        ('independent_max', 'strong'): '#009E73',
        ('one_to_one', 'weak'): '#eb6834',
        ('one_to_one', 'strong'): '#AA4499',
    }
    if semantic.get('reach_definition')!='reference_ancestor_closure_v1':
        raise ValueError('Expected ancestor-closed reach calibration')
    data=[]; summary=[]
    for (rid,tau,policy),s in scores.items():
        if policy not in POLICIES:continue
        c=wide[rid]['corpus'];cal=semantic['reach_calibration'][c]
        reach=set(cal['reach_features']);matched=set(json.loads(s['matched_reference_ids']))
        if len(reach)!=cal['reach_size'] or len(matched)!=int(s['n_reference_matched']):
            raise ValueError('Reach or matched-reference cardinality mismatch')
        if int(s['n_reference'])!=cal['F_t_size']:raise ValueError('Reference size mismatch')
        outside=matched-reach
        data.append(dict(run_id=rid,corpus=c,tau=tau,policy=policy,
            n_matched=len(matched),reach_size=len(reach),out_of_reach_count=len(outside),
            out_of_reach_ids=json.dumps(sorted(outside)),weak=len(matched)>len(reach),strong=bool(outside)))
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.15),sharey=True,layout='constrained')
    for ax,c in zip(axes,['repair','federation']):
        for p,policy in enumerate(POLICIES):
            for test,marker in [('weak','o'),('strong','^')]:
                ys=[]
                for tau in [.3,.4,.5,.6]:
                    rs=[r for r in data if r['corpus']==c and r['policy']==policy and r['tau']==tau]
                    if not rs:raise ValueError(f'Missing closure cell {c,policy,tau}')
                    pct=100*sum(r[test] for r in rs)/len(rs);ys.append(pct)
                    summary.append(dict(corpus=c,policy=policy,test=test,tau=tau,n=len(rs),percent=pct))
                ax.plot([.3,.4,.5,.6],ys,color=series_colors[policy,test],ls=['-','--'][p],marker=marker,ms=4,
                        label=f'{"Independent" if p==0 else "One-to-one"}: {test}')
        ax.set(title=c.title(),xlabel='Similarity threshold τ',ylim=(-2,103),xticks=[.3,.4,.5,.6])
    axes[0].set_ylabel('Runs flagged (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,fontsize=7,frameon=False)
    fig.get_layout_engine().set(rect=(0,0,1,.77))
    save(fig,out,'fig_closure_tau',data)
    return dict(cells=summary,note='Weak: full recall exceeds annotated closed-reach ratio. Strong: any matched reference node outside closed reach. Not proof of hallucination or theorem falsification.')


def variability(wide,out):
    family_path=REPO/'config/analysis/families/C_ablation.json'
    family=json.loads(family_path.read_text())
    data=[]
    for model in MODELS:
        for arm in ['guided','ablated']:
            selectors=[c['arm_a' if arm=='guided' else 'arm_b'] for c in family['comparisons']]
            rs=[w for w in wide.values() if w['model_id']==model
                and any(all(str(w[k]).lower()==str(v).lower() for k,v in s.items()) for s in selectors)
                and w['structural__structural_conformance__status']=='ok' and yes(w['structural__structural_conformance'])]
            def vals(key):
                k='structural__'+key
                return [float(r[k]) for r in rs if r[k+'__status']=='ok' and r[k]!='']
            counts=[sum(vals('n_'+kind)) for kind in ['and','or','alt']];total=sum(counts)
            mandatory=vals('mandatory_ratio')
            deg=[yes(r['structural__degenerate']) for r in rs if r['structural__degenerate__status']=='ok']
            data.append(dict(model_id=model,arm=arm,n=len(rs),total_groups=total,
                **{kind+'_share':count/total if total else None for kind,count in zip(['and','or','alt'],counts)},
                median_mandatory=st.median(mandatory) if mandatory else None,n_mandatory=len(mandatory),
                run_ids=json.dumps(sorted(r['run_id'] for r in rs)),
                degenerate_percent=100*st.mean(deg) if deg else None,n_degenerate=len(deg)))
    fig,ax=plt.subplots(figsize=(WIDTH,2.8),layout='constrained')
    for y,r in enumerate(data):
        left=0
        for kind,col,hatch in zip(['and','or','alt'],['#b9bdc3','#2a78d6','#eb6834'],['','///','xx']):
            v=r[kind+'_share']
            if v is None:continue
            ax.barh(y,100*v,left=left,color=col,hatch=hatch,edgecolor='white',height=.65,label=kind.upper() if y==0 else None)
            left+=100*v
        text=(f'M={r["median_mandatory"]:.2f}; D={r["degenerate_percent"]:.0f}%' if r['median_mandatory'] is not None and r['degenerate_percent'] is not None else 'unavailable')
        ax.text(1.02,y,text,transform=ax.get_yaxis_transform(),clip_on=False,va='center',fontsize=7)
    ax.set_yticks(range(len(data)),[f'{NAMES[MODELS.index(r["model_id"])]} {r["arm"]}\n(n={r["n"]})' for r in data])
    ax.set(xlim=(0,100),xlabel='Group share (%)')
    ax.set_xticks([0,25,50,75,100]);ax.invert_yaxis();ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18))
    save(fig,out,'fig_variability',data)
    return dict(family_path=str(family_path),family_sha256=sha(family_path),
                caption='Conformant outputs from the declared Family C configuration selectors. M: median mandatory ratio; D: percentage of eligible runs flagged degenerate. Group shares pool AND/OR/ALT counts.',
                note='Local structural_conformance=True, not strict admissibility or FeatureIDE acceptance. RAG, primary, k_doc=5; open models N1/N10, Astra N10. Unequal surviving cell mixtures: pooled descriptive comparison, not the Family C hypothesis tests.')


def outcomes(wide,inventory,out):
    categories=['completed','truncated_output','malformed_xml','empty_response']
    data=[]
    for model in MODELS:
        for grounding in ['rag','nonrag']:
            rs=[r for r in wide.values() if r['model_id']==model and r['grounding']==grounding and guided(r)]
            counts=Counter(inventory[r['run_id']]['recorded_terminal_status'] for r in rs)
            if set(counts)-set(categories):raise ValueError(f'Unrepresented outcome categories: {counts}')
            data.append(dict(model_id=model,grounding=grounding,planned=len(rs),**{c:counts[c] for c in categories}))
    fig,ax=plt.subplots(figsize=(WIDTH*.75,2.9),layout='constrained')
    bottom=np.zeros(len(data))
    for c,col,hatch,label in zip(categories,['#1baf7a','#eb6834','#2a78d6','#777777'],['','///','xx','..'],['Completed','Truncated','Malformed','Empty']):
        v=np.array([100*r[c]/r['planned'] if r['planned'] else 0 for r in data])
        ax.bar(range(len(data)),v,bottom=bottom,color=col,hatch=hatch,label=label if c!='empty_response' else '_nolegend_',edgecolor='white');bottom+=v
    ax.set_xticks(range(len(data)),[f'{NAMES[MODELS.index(r["model_id"])]}\n{"RAG" if r["grounding"]=="rag" else "non-RAG"}\nn={r["planned"]}' for r in data],fontsize=6)
    ax.set(ylabel='Planned runs (%)',ylim=(0,100))
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7)
    fig.get_layout_engine().set(rect=(0,0,1,.82))
    save(fig,out,'fig_outcomes',data)
    empty=sum(r['empty_response'] for r in data)
    return dict(empty_response_count=empty,
                caption=f'Guided-arm outcomes as percentages of planned runs. Includes {empty} empty-response run(s), retained in the stacks but omitted from the legend.',
                note='Guided arms only; heterogeneous configurations. Failure frequencies do not establish output-size causation. Completed does not imply conformance.')


def ordering(wide,scores,out):
    data=[]; cells=[]
    fig,ax=plt.subplots(figsize=(WIDTH*.6,2.6),layout='constrained');rng=np.random.default_rng(20261001)
    for model in MODELS[:2]:
        for order in ['primary','alt_1','alt_2']:
            rs=[r for r in wide.values() if r['corpus']=='repair' and int(r['N'])==10 and r['grounding']=='rag'
                and r['model_id']==model and r['ordering_id']==order and int(r['k_doc'])==5 and yes(r['metamodel_block'])
                and r['arm']==('guided_headline' if order=='primary' else 'order_sensitivity')]
            vals=[];x=len(cells)
            for r in rs:
                v=score(scores,r['run_id'],'one_to_one')
                if v is not None:
                    vals.append(v);data.append(dict(run_id=r['run_id'],model_id=model,ordering=order,f1=v))
            mean=st.mean(vals) if vals else None;sd=st.stdev(vals) if len(vals)>1 else None
            cells.append(dict(model_id=model,ordering=order,planned=len(rs),n=len(vals),mean=mean,sd=sd))
            idx=MODELS.index(model)
            ax.scatter(x+rng.uniform(-.16,.16,len(vals)),vals,c=COLORS[idx],marker=MARKERS[idx],s=15,zorder=3)
            if sd is not None:ax.fill_between([x-.3,x+.3],mean-sd,mean+sd,color=COLORS[idx],alpha=.15)
            if mean is not None:ax.plot([x-.23,x+.23],[mean,mean],color='black',lw=1)
    ax.set_xticks(range(len(cells)),[f'{r["ordering"]}\n{r["n"]}/{r["planned"]}' for r in cells],fontsize=7)
    if any(not .2<=r['f1']<=.6 for r in data):
        raise ValueError('Ordering score outside requested 0.2–0.6 axis; widen limits before plotting')
    ax.set(ylabel='One-to-one F1',xlabel='Ordering (scored/planned)',ylim=(.2,.6))
    fig.legend([Line2D([],[],color=COLORS[i],marker=MARKERS[i],ls='') for i in range(2)],
               NAMES[:2],loc='upper center',ncol=2,frameon=False,fontsize=7)
    fig.get_layout_engine().set(rect=(0,0,1,.9))
    save(fig,out,'fig_ordering',data)
    return dict(cells=cells,note='Band = mean ± sample SD, not CI. Primary uses all headline repetitions; alternate orderings may have fewer survivors. No small-n significance claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,default=BASE)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output directory')
    paths={k:a.analysis/v for k,v in dict(wide='aggregate-current-v4/wide.csv',
        inventory='inventory-current-v1/runs.csv',tau='tau-matching-v2/tau_sweep.csv',
        semantic='semantic-current-v3/summary.json').items()}
    wide=unique(read(paths['wide']),lambda r:r['run_id'])
    inventory=unique(read(paths['inventory']),lambda r:r['run_id'])
    scores=unique(read(paths['tau']),lambda r:(r['run_id'],float(r['tau']),r['matching_policy']))
    if set(wide)!=set(inventory):raise ValueError('Inventory and aggregate run populations differ')
    if set(r['model_id'] for r in wide.values())-set(MODELS):raise ValueError('Unknown model palette')
    completed={rid for rid,r in inventory.items() if yes(r['completed'])}
    for tau in [.3,.4,.5,.6]:
        for policy in POLICIES:
            ids={rid for rid,t,p in scores if t==tau and p==policy}
            if ids!=completed:raise ValueError(f'Incomplete scored population: {tau,policy}')
    semantic=json.loads(paths['semantic'].read_text())
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.titlesize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':2,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'axes.grid.axis':'y','grid.color':'#e4e4e4','grid.linewidth':.5,'axes.axisbelow':True,
        'pdf.fonttype':42})
    a.output.mkdir(parents=True)
    notes={}
    notes['inflation']=inflation(wide,scores,a.output)
    notes['granularity']=granularity(wide,scores,a.output)
    notes['closure']=closure(wide,scores,semantic,a.output)
    notes['variability']=variability(wide,a.output)
    notes['outcomes']=outcomes(wide,inventory,a.output)
    notes['ordering']=ordering(wide,scores,a.output)
    manifest=dict(inputs={k:dict(path=str(v.resolve()),sha256=sha(v)) for k,v in paths.items()},
        script_sha256=sha(Path(__file__)),notes=notes,matplotlib_version=matplotlib.__version__,
        outputs={f.name:sha(f) for f in sorted(a.output.iterdir())})
    (a.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved six PDFs, PNG previews, source CSVs and figure_manifest.json in {a.output}')
    print('Granularity y-axis:',notes['granularity']['ylim'])


if __name__=='__main__':main()
