"""Protocol-v3 selection, stratified citation preparation, and blinded packets."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import zipfile

from openpyxl import load_workbook
from scripts import prepare_expert_artifacts as common
from fame.evaluation.alignment import extract_indexed_nodes
from fame.evaluation.reachable import load_attribution
from fame.evaluation.sibling_agreement import unpack_pairs, correspondence
from scripts.tau_rescore import _iter_run_pairs

REPO=common.REPO
CELLS=[('deepseek-v4.1-flash:cloud',1),('glm-5.3-flash:cloud',1),
       ('deepseek-v4.1-flash:cloud',10),('glm-5.3-flash:cloud',10),('gpt-6-astra',1)]
CODES={'repair':['Rep-96','Rep-30','Rep-28','Rep-57','Rep-79'],
       'federation':['Fed-36','Fed-11','Fed-97','Fed-67','Fed-44']}
EDIT_FIELDS=('excerpt','excerpt_origin','excerpt_locator','source_checked_by')


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows:raise ValueError('Refusing empty table')
    common._write_csv(path,list(rows[0]),rows)


def fresh(path):
    if path.exists():raise FileExistsError(f'Refusing overwrite: {path}')


def freeze_files(folder,names):
    (folder/'hashes.json').write_text(json.dumps({n:common._sha256(folder/n) for n in names},indent=2)+'\n')


def check_frozen(folder):
    for name,digest in common._read_json(folder/'hashes.json').items():
        if common._sha256(folder/name)!=digest:raise ValueError(f'Frozen file changed: {folder/name}')


def select(args):
    fresh(args.output)
    attempts=common._astra_attempts(args.astra_results)
    paths=[]
    for row in common._read_json(args.open_inventory):
        if (row.get('arm') in ('guided_baseline','guided_headline')
            and row.get('model_id') in [c[0] for c in CELLS[:2]]
            and row.get('N') in (1,10) and row.get('seed') in range(5)
            and row.get('grounding')=='rag' and row.get('ordering_id')=='primary'
            and row.get('k_doc')==5 and row.get('metamodel_block') is True):
            paths.append((REPO/row['resolved_path']/'run_meta.json',row['run_id']))
    paths.extend((Path(a['run_meta_path']),a['run_id']) for a in attempts)
    candidates={};excluded=[];seen=set()
    for path,rid in paths:
        m=common._read_json(path);c=m['config']
        key=(c['corpus'],int(c['seed']),c['model_id'],int(c['N']))
        if key in seen:raise ValueError(f'Duplicate candidate cell: {key}')
        seen.add(key)
        if m['run_id']!=rid:raise ValueError('Inventory identity mismatch')
        xml=path.parent/'fm_gen.xml'
        if not m.get('completed') or m.get('terminal_status')!='completed':
            excluded.append(dict(run_id=rid,reason='incomplete'));continue
        try:
            tree=common._parse_xml(xml)
            if tree.docinfo.doctype:raise ValueError('DTD unsupported')
            if tree.getroot().tag!='featureModel' or len(tree.findall('struct'))!=1 or len(tree.find('struct'))!=1:
                raise ValueError('Expected one feature-model tree')
            nodes=extract_indexed_nodes(xml)
            if not nodes:raise ValueError('Empty feature tree')
        except (OSError,ValueError,common.etree.XMLSyntaxError) as e:
            excluded.append(dict(run_id=rid,reason=str(e)));continue
        candidates[key]=dict(run_id=rid,corpus=c['corpus'],seed=c['seed'],N=c['N'],model_id=c['model_id'],
            model_key=common.MODELS[c['model_id']],xml_path=str(xml.resolve()),xml_sha256=common._sha256(xml),
            run_meta_path=str(path.resolve()),config=c)
    items=[];policy={}
    for corpus in common.CORPORA:
        eligible=[s for s in range(5) if all((corpus,s,m,n) in candidates for m,n in CELLS)]
        if not eligible:raise ValueError(f'No complete five-cell set for {corpus}')
        seed=eligible[0];policy[corpus]=dict(eligible=eligible,selected=seed)
        group=[dict(candidates[corpus,seed,m,n],code=code) for (m,n),code in zip(CELLS,CODES[corpus])]
        for field in common.MATCH_FIELDS:
            if field=='N':continue
            if len({json.dumps(i['config'].get(field),sort_keys=True) for i in group})!=1:
                raise ValueError(f'Configuration mismatch: {corpus} {field}')
        items.extend(group)
    args.output.mkdir(parents=True)
    selection=dict(version=3,blind_seed=20261002,items=items,seed_policy=policy,excluded=excluded,
        code_policy='Author-approved codes from protocol v3 table; no code redraw',
        open_inventory_sha256=common._sha256(args.open_inventory),
        protocol_sha256=common._sha256(REPO/'docs/expert-evaluation-strategy-v3.md'),
        implementation_sha256=common._sha256(Path(__file__)))
    (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    write_csv(args.output/'expert_sample_frozen.csv',[{k:i[k] for k in ('code','corpus','model_id','N','seed','run_id','xml_path','xml_sha256')} for i in items])
    write_csv(args.output/'astra_attempts.csv',attempts)
    freeze_files(args.output,['selection.json','expert_sample_frozen.csv','astra_attempts.csv'])
    print(f'Selected 10 models; author-only key: {args.output}')


def load_selection(folder):
    check_frozen(folder)
    s=common._read_json(folder/'selection.json')
    if s['version']!=3 or len(s['items'])!=10:raise ValueError('Requires a v3 ten-model selection')
    if len({i['run_id'] for i in s['items']})!=10 or len({i['code'] for i in s['items']})!=10:
        raise ValueError('Duplicate run/code')
    for i in s['items']:
        if common._sha256(Path(i['xml_path']))!=i['xml_sha256']:raise ValueError('Selected XML changed')
    return s


def sample_edges(pool,seed=20261002):
    """Uniform independent 3-subset draws, conditioned jointly on per-model cap."""
    rng=random.Random(seed);chosen=[]
    for corpus in common.CORPORA:
        strata=[[r for r in pool if r['corpus']==corpus and r['stratum']==s] for s in ('agree','disagree')]
        if any(len(rs)<3 for rs in strata):raise ValueError(f'Insufficient citation stratum: {corpus}')
        for _ in range(100000):
            draw=rng.sample(strata[0],3)+rng.sample(strata[1],3)
            if max(Counter(r['run_id'] for r in draw).values())<=2:
                chosen.extend(draw);break
        else:raise ValueError(f'No valid constrained draw found for {corpus}; do not relax quotas silently')
    rng.shuffle(chosen)
    return [dict(r,item_id=f'C{k:02d}') for k,r in enumerate(chosen,1)]


def citations(args):
    fresh(args.output);selection=load_selection(args.selection_dir)
    wanted={i['run_id']:i for i in selection['items']};by_run={}
    for path in [args.pairs]+([args.astra_pairs] if args.astra_pairs else []):
        for pairs in _iter_run_pairs(path):
            rid=pairs[0]['run_id']
            if rid in wanted:
                if rid in by_run:raise ValueError('Duplicate saved run pairs')
                by_run[rid]=pairs
    missing=set(wanted)-set(by_run)
    if any(wanted[r]['model_id']!='gpt-6-astra' for r in missing):raise ValueError('Selected campaign run missing from pairs')
    encoder_identity=None
    if missing:
        from scripts import evaluate_semantic as sem
        print('Computing selected Astra similarities locally (no API calls).',flush=True)
        encoder,encoder_identity=sem._encoder_or_die(sem._load_contract())
        for rid in sorted(missing):
            i=wanted[rid];c=i['corpus'];info=sem._partition_and_reach_for(c)
            outcome=sem.evaluate_semantic(Path(i['xml_path']),sem.GROUND_TRUTH[c],encoder=encoder,
                tau_primary=.4,tau_sweep=[.4],attested=info['partition'].attested,
                organising=info['partition'].organising,reach=info['reach'],rho=info['rho'],
                attribution=load_attribution(sem.ATTRIBUTION[c]))
            if not outcome['pairs']:raise ValueError(f'No usable similarities for {rid}')
            by_run[rid]=outcome['pairs']
    docs=common._doc_index();pool=[]
    for rid,item in wanted.items():
        matrix,g,r=unpack_pairs(by_run[rid])
        xmlnodes=extract_indexed_nodes(Path(item['xml_path']))
        refnodes=extract_indexed_nodes(REPO/f'data/ground_truth/{item["corpus"]}.xml')
        for old,new in [(g,xmlnodes),(r,refnodes)]:
            if [(n['name'],n['parent_index']) for n in old]!=[(n['name'],n['parent_index']) for n in new]:
                raise ValueError('Saved pairs do not match selected XML/reference topology')
        mapping=correspondence(matrix,g,r,'independent_max',.4)
        attr=load_attribution(REPO/f'data/attribution/{item["corpus"]}.csv')
        for gi,rj in sorted(mapping.items()):
            refname=r[rj]['name']
            if not attr.get(refname):continue
            names=[];at=gi
            while at is not None:names.append(g[at]['name']);at=g[at]['parent_index']
            path=' / '.join(reversed(names))
            for doc in sorted(set(xmlnodes[gi]['citations'])):
                if doc not in docs or not doc.startswith('rep_' if item['corpus']=='repair' else 'fed_'):continue
                pool.append(dict(run_id=rid,code=item['code'],corpus=item['corpus'],gen_index=gi,
                    feature=path,doc_id=doc,doc_title=docs[doc].get('title',''),matched_reference=refname,
                    similarity=float(matrix[gi,rj]),stratum='agree' if doc in attr[refname] else 'disagree'))
    pool.sort(key=lambda r:(r['corpus'],r['run_id'],r['gen_index'],r['doc_id']))
    sampled=sample_edges(pool)
    args.output.mkdir(parents=True)
    write_csv(args.output/'citation_candidates_AUTHOR_ONLY.csv',pool)
    write_csv(args.output/'citation_key_AUTHOR_ONLY.csv',sampled)
    blinded=[dict(item_id=r['item_id'],code=r['code'],corpus=r['corpus'],feature=r['feature'],
                  doc_id=r['doc_id'],doc_title=r['doc_title'],**{k:'' for k in EDIT_FIELDS}) for r in sampled]
    write_csv(args.output/'citation_items_to_complete.csv',blinded)
    summary=dict(version=3,seed=20261002,tau=.4,pool_size=len(pool),selection_sha256=common._sha256(args.selection_dir/'selection.json'),
        pairs_sha256=common._sha256(args.pairs),astra_pairs_sha256=common._sha256(args.astra_pairs) if args.astra_pairs else None,
        encoder_identity=encoder_identity,implementation_sha256=common._sha256(Path(__file__)),
        sampling='Uniform 3-subsets per corpus/stratum; rejection conditions on max two edges per selected model.',
        reference_sha256={c:common._sha256(REPO/f'data/ground_truth/{c}.xml') for c in common.CORPORA},
        attribution_sha256={c:common._sha256(REPO/f'data/attribution/{c}.csv') for c in common.CORPORA})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    freeze_files(args.output,['citation_candidates_AUTHOR_ONLY.csv','citation_key_AUTHOR_ONLY.csv','citation_items_to_complete.csv','summary.json'])
    print(f'Give ONLY citation_items_to_complete.csv to the excerpt preparer; keep author-only files hidden. Output: {args.output}')


def validated_citations(args,selection):
    check_frozen(args.citation_dir)
    summary=common._read_json(args.citation_dir/'summary.json')
    if summary['selection_sha256']!=common._sha256(args.selection_dir/'selection.json'):raise ValueError('Citation selection mismatch')
    frozen=read_csv(args.citation_dir/'citation_items_to_complete.csv');final=read_csv(args.citation_items)
    if len(final)!=12 or len({r['item_id'] for r in final})!=12:raise ValueError('Expected 12 unique citation items')
    base={r['item_id']:r for r in frozen}
    for r in final:
        if r['item_id'] not in base:raise ValueError('Unknown citation identity')
        if set(r)!=set(base[r['item_id']]):raise ValueError('Unexpected citation columns')
        for k,v in base[r['item_id']].items():
            if k not in EDIT_FIELDS and r[k]!=v:raise ValueError(f'Frozen citation identity changed: {k}')
        if any(not r[k].strip() for k in EDIT_FIELDS):raise ValueError('Complete excerpts, origin, locator and source_checked_by')
        if r['excerpt_origin'] not in ('retrieved chunk','document excerpt'):raise ValueError('Invalid excerpt origin')
    return final


def fill_workbook(wb,items,citations,rater):
    expected=['Read me first','Rating guide','1. Rate the models','2. Rank within domain','3. Check citations']
    if wb.sheetnames!=expected:raise ValueError('Use the v3 workbook, with no additional/hidden sheets')
    if any(s.sheet_state!='visible' for s in wb):raise ValueError('Hidden sheets not allowed')
    if wb[expected[2]].max_row!=11 or wb[expected[4]].max_row!=13:raise ValueError('Wrong v3 row counts')
    # Blank responses are mandatory; never silently erase an expert's work.
    for sheet,coords in [(expected[2],[(r,c) for r in range(2,12) for c in range(3,13)]),
                         (expected[4],[(r,c) for r in range(2,14) for c in (7,8)]),
                         (expected[3],[(r,4) for r in list(range(3,10))+list(range(13,20))])]:
        if any(wb[sheet].cell(r,c).value not in (None,'') for r,c in coords):raise ValueError('Workbook already contains responses')
    rng=random.Random(20261002+rater*101)
    corpora=list(common.CORPORA) if rater%2 else list(reversed(common.CORPORA))
    order=[]
    for block,c in enumerate(corpora):
        group=[i for i in items if i['corpus']==c];rng.shuffle(group);order.extend(group)
        ws=wb[expected[3]];offset=block*10
        ws.cell(1+offset,4,c.title())
        for row,item in enumerate(group,3+offset):ws.cell(row,3,item['code'])
    ws=wb[expected[2]]
    for row,item in enumerate(order,2):ws.cell(row,1,item['code']);ws.cell(row,2,item['corpus'].title())
    shuffled=citations[:];rng.shuffle(shuffled)
    for row,item in enumerate(shuffled,2):
        vals=[item['item_id'],item['code'],item['corpus'].title(),item['feature'],
              f'{item["doc_id"]} — {item["doc_title"]}',
              f'{item["excerpt"]}\n[{item["excerpt_origin"]}: {item["excerpt_locator"]}]']
        for col,value in enumerate(vals,1):wb[expected[4]].cell(row,col,value)
    wb.properties.creator='Study team';wb.properties.lastModifiedBy='Study team'
    wb.properties.title=f'Expert evaluation R{rater:02d}'
    for sheet in wb:
        for row in sheet:
            for cell in row:
                # Strip comment-author metadata from the packet copy; keep source intact.
                if cell.comment:cell.comment=None
    return order


def pack(args):
    fresh(args.output);selection=load_selection(args.selection_dir);items=selection['items']
    rows=validated_citations(args,selection)
    # Validate template before writing any packets.
    fill_workbook(load_workbook(args.form),items,rows,1)
    args.output.mkdir(parents=True);author=args.output/'AUTHOR_ONLY';author.mkdir()
    write_csv(author/'code_key.csv',[{k:i[k] for k in ('code','run_id','corpus','model_id','N','seed')} for i in items])
    key=read_csv(args.citation_dir/'citation_key_AUTHOR_ONLY.csv');write_csv(author/'citation_key.csv',key)
    orders=[];hashes=[];docs=common._doc_index()
    for rater in range(1,args.raters+1):
        code=f'R{rater:02d}';folder=args.output/code;models=folder/'models';models.mkdir(parents=True)
        wb=load_workbook(args.form);order=fill_workbook(wb,items,rows,rater)
        for item in order:
            page=common._render_html(item,item['code'])
            # Common renderer removes trace markers; fail on visible study identifiers.
            if any(x in page for x in list(common.MODELS)+[i['run_id'] for i in items]):raise ValueError('Identity leaked in model text; review source')
            (models/(item['code']+'.html')).write_text(page,encoding='utf-8')
        wb.save(folder/f'Expert-rating-{code}.xlsx')
        write_csv(folder/'source-index.csv',[dict(doc_id=d,title=v.get('title',''),doi_url=v.get('doi_url','')) for d,v in sorted(docs.items())])
        (folder/'README.txt').write_text('Complete Task 1, then the two within-domain rankings, then 12 citation items. Open models/<code>.html. Do not infer quality from size alone. Use NJ/Cannot judge where needed. Authorized source access is supplied separately. Work independently.\n')
        zip_path=args.output/(code+'.zip')
        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(folder.rglob('*')):
                if file.is_file():z.write(file,file.relative_to(folder))
        hashes.append(dict(rater=code,sha256=common._sha256(zip_path)))
        orders.append(dict(rater=code,rating_codes=[i['code'] for i in order],citation_ids=[wb['3. Check citations'].cell(r,1).value for r in range(2,14)]))
    write_csv(author/'packet_hashes.csv',hashes)
    (author/'build.json').write_text(json.dumps(dict(orders=orders,form_sha256=common._sha256(args.form),
        selection_sha256=common._sha256(args.selection_dir/'selection.json'),citation_items_sha256=common._sha256(args.citation_items),
        implementation_sha256=common._sha256(Path(__file__)),renderer_sha256=common._sha256(Path(common.__file__))),indent=2)+'\n')
    print(f'Created {args.raters} v3 packets. Keep AUTHOR_ONLY private; pilot and inspect before dispatch.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('select',help='Freeze ten models at lowest complete five-cell seed per corpus')
    s.add_argument('--open-inventory',type=Path,default=REPO/'results/ifs-2027/analysis/inventory-current-v1/runs.json')
    s.add_argument('--astra-results',type=Path,default=REPO/'results/expert-astra-n1-rag-v1')
    s.add_argument('--output',type=Path,required=True);s.set_defaults(func=select)
    c=sub.add_parser('citations',help='Draw 12 stratified edges; selected Astra similarities may be computed locally')
    c.add_argument('--selection-dir',type=Path,required=True)
    c.add_argument('--pairs',type=Path,default=REPO/'results/ifs-2027/analysis/semantic-current-v3/pairs.csv')
    c.add_argument('--astra-pairs',type=Path,help='Optional saved raw pairs for selected extension runs')
    c.add_argument('--output',type=Path,required=True);c.set_defaults(func=citations)
    b=sub.add_parser('pack',help='Build v3 workbooks: 10 ratings, 2 rankings, 12 citations')
    b.add_argument('--selection-dir',type=Path,required=True);b.add_argument('--citation-dir',type=Path,required=True)
    b.add_argument('--citation-items',type=Path,required=True)
    b.add_argument('--form',type=Path,default=REPO/'data/Expert-evaluation-form-v3.xlsx')
    b.add_argument('--raters',type=int,choices=[2,3],default=3)
    b.add_argument('--output',type=Path,required=True);b.set_defaults(func=pack)
    args=p.parse_args();args.func(args);return 0
