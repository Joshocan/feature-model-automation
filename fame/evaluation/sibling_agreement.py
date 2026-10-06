"""Post-generation sibling diagnostics, conditional on matched non-root nodes.

Parent labels are not compared when scoring siblings. Child correspondences
still depend on the declared name/embedding matching policy.
"""
from collections import Counter
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def unpack_pairs(rows):
    """Validate a complete occurrence-indexed matrix and saved tree metadata."""
    generated, reference, values = {}, {}, {}
    for row in rows:
        ids = []
        for prefix, target in [('gen', generated), ('ref', reference)]:
            index = int(row[prefix + '_index'])
            raw = row[prefix + '_parent_index']
            parent = None if raw in ('', None) else int(raw)
            node = dict(name=row[prefix + '_name'], parent_index=parent)
            if index in target and target[index] != node:
                raise ValueError('Inconsistent occurrence metadata')
            target[index] = node
            ids.append(index)
        key = tuple(ids)
        value = float(row['similarity'])
        if key in values or not math.isfinite(value) or not -1.00001 <= value <= 1.00001:
            raise ValueError('Duplicate pair or invalid similarity')
        values[key] = value
    for nodes in (generated, reference):
        if not nodes or sorted(nodes) != list(range(len(nodes))):
            raise ValueError('Expected contiguous occurrence indices')
        if sum(n['parent_index'] is None for n in nodes.values()) != 1:
            raise ValueError('Expected one tree root')
        for i, node in nodes.items():
            parent = node['parent_index']
            if parent is not None and (parent not in nodes or parent >= i):
                raise ValueError('Invalid preorder parent index')
    if len(values) != len(generated) * len(reference):
        raise ValueError('Incomplete similarity matrix')
    matrix = np.array([[values[i, j] for j in range(len(reference))]
                       for i in range(len(generated))])
    return matrix, list(generated[i] for i in range(len(generated))), list(reference[i] for i in range(len(reference)))


def correspondence(matrix, generated, reference, policy, tau):
    if policy == 'exact_unique':
        gc = Counter(n['name'] for n in generated)
        rc = Counter(n['name'] for n in reference)
        refs = {n['name']: j for j, n in enumerate(reference) if rc[n['name']] == 1}
        return {i: refs[n['name']] for i, n in enumerate(generated)
                if gc[n['name']] == 1 and n['name'] in refs}
    if policy == 'independent_max':
        return {i: int(np.argmax(row)) for i, row in enumerate(matrix) if row.max() >= tau}
    if policy != 'one_to_one':
        raise ValueError('Unknown matching policy')
    bonus = 2 * min(matrix.shape) + 1
    weights = np.zeros((len(generated), len(reference) + len(generated)))
    weights[:, :len(reference)] = np.where(matrix >= tau, bonus + matrix, -1e12)
    gi, rj = linear_sum_assignment(weights, maximize=True)
    return {int(i): int(j) for i, j in zip(gi, rj) if j < len(reference) and matrix[i, j] >= tau}


def sibling_metrics(generated, reference, mapping):
    # Roots excluded on both sides; two nodes mapping to the SAME reference
    # occurrence cannot count as a reference sibling pair.
    eligible = [(i, j) for i, j in mapping.items()
                if generated[i]['parent_index'] is not None and reference[j]['parent_index'] is not None]
    def pairs(keys):
        return sum(n * (n - 1) // 2 for n in Counter(keys).values())
    gp = lambda i: generated[i]['parent_index']
    rp = lambda j: reference[j]['parent_index']
    collision = pairs(j for i, j in eligible)
    same_group_collision = pairs((gp(i), j) for i, j in eligible)
    predicted = pairs(gp(i) for i, j in eligible) - same_group_collision
    expected = pairs(rp(j) for i, j in eligible) - collision
    tp = pairs((gp(i), rp(j)) for i, j in eligible) - same_group_collision
    evaluated = [(i, j) for i, j in eligible if gp(i) in mapping]
    correct = sum(mapping[gp(i)] == rp(j) for i, j in evaluated)
    return dict(n_generated=len(generated), n_reference=len(reference),
                n_matched_generated=len(mapping), n_matched_reference=len(set(mapping.values())),
                n_eligible_nodes=len(eligible),
                n_eligible_pairs=len(eligible)*(len(eligible)-1)//2-collision,
                n_same_reference_pairs_excluded=collision,
                sibling_tp=tp, sibling_fp=predicted-tp, sibling_fn=expected-tp,
                n_generated_sibling_pairs=predicted, n_reference_sibling_pairs=expected,
                sibling_precision=tp/predicted if predicted else None,
                sibling_recall=tp/expected if expected else None,
                sibling_f1=2*tp/(predicted+expected) if predicted+expected else None,
                sibling_precision_status='ok' if predicted else 'not_applicable',
                sibling_recall_status='ok' if expected else 'not_applicable',
                sibling_f1_status='ok' if predicted+expected else 'not_applicable',
                n_parent_evaluable=len(evaluated), n_parent_correct=correct,
                parent_match_rate=correct/len(evaluated) if evaluated else None)
