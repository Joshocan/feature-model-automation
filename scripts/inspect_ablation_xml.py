#!/usr/bin/env python3
"""Inspect first two completed seeds per DeepSeek Repair N10 RAG arm."""
import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

TAGS={'and','or','alt','feature'}
PHRASES=['exactly one feature has no parent','every feature name is unique',
         'exactly two operands','every non-root feature has exactly one parent',
         'the parent relation is acyclic','children of the same parent',
         'the output must validate against this schema','independently selectable']


def inspect(path):
    root=ET.parse(path).getroot();counts=Counter();depths=[];mandatory=[]
    def walk(node,depth):
        if node.tag not in TAGS:
            return
        counts[node.tag]+=1;depths.append(depth)
        if depth>0 and node.get('mandatory')=='true':
            mandatory.append(node.get('name'))
        for child in node:
            walk(child,depth+1)
    for node in root.find('struct'):
        walk(node,0)
    text=' '.join(' '.join(n.itertext()) for n in root.iter('description'))
    normalized=re.sub(r'\s+',' ',text).lower()
    groups=sum(counts[t] for t in ('and','or','alt'))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                counts={t:counts[t] for t in sorted(TAGS)},
                group_proportions={t:counts[t]/groups if groups else None for t in ('and','or','alt')},
                nonroot_mandatory=len(mandatory),mandatory_names=mandatory,
                max_depth=max(depths),phrase_hits=[p for p in PHRASES if p in normalized])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,default=Path('results/ifs-2027/analysis/inventory-current-v1/runs.csv'))
    args=p.parse_args()
    with args.inventory.open() as f: rows=list(csv.DictReader(f))
    output=[]
    for arm in ('guided_headline','ablation'):
        selected=sorted([r for r in rows if r['arm']==arm and r['model_id']=='deepseek-v4.1-flash:cloud'
                         and r['corpus']=='repair' and r['N']=='10' and r['grounding']=='rag'
                         and r['inventory_status']=='completed'],
                        key=lambda r:(int(r['seed']),int(r['repetition']),r['run_id']))[:2]
        if len(selected)!=2: raise ValueError('Expected two completed outputs per arm')
        for r in selected:
            output.append(dict(arm=arm,seed=r['seed'],run_id=r['run_id'],
                               **inspect(Path(r['resolved_path'])/'fm_gen.xml')))
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
