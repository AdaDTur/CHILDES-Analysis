import os
from collections import Counter
from nltk.tree import Tree

def load_trees_from_file(filepath):
    trees = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tree = Tree.fromstring(line)
                trees.append(tree)
            except Exception as e:
              pass
    return trees

def compute_mlu_words(trees):
    total_words = 0
    for t in trees:
        total_words += len(t.leaves())
    mlu = total_words / len(trees) if trees else 0.0
    return mlu

def compute_mlu_morphemes(trees, morphological_rules=None):
    if morphological_rules is None:
        morphological_rules = lambda token: 1

    total_morphemes = 0
    for t in trees:
        tokens = t.leaves()
        for token in tokens:
            total_morphemes += morphological_rules(token)
    mlu_morph = total_morphemes / len(trees) if trees else 0.0
    return mlu_morph

def compute_average_tree_depth(trees):
    if not trees:
        return 0.0
    total_depth = sum(t.height() for t in trees)
    return total_depth / len(trees)

def compute_subordinate_clause_stats(trees):
    sbar_count = 0
    utterances_with_sbar = 0

    for t in trees:
        found_sbar = False
        for subtree in t.subtrees():
            if subtree.label() == 'SBAR':
                sbar_count += 1
                found_sbar = True
        if found_sbar:
            utterances_with_sbar += 1

    total_utterances = len(trees)
    return {
        "total_sbar": sbar_count,
        "utterances_with_sbar": utterances_with_sbar,
        "proportion_of_utterances_with_sbar": (
            utterances_with_sbar / total_utterances if total_utterances else 0.0
        )
    }

def compute_pos_frequencies(trees):
    pos_counter = Counter()

    for t in trees:
        for subtree in t.subtrees(lambda st: st.height() == 2):
            pos_tag = subtree.label()
            pos_counter[pos_tag] += 1

    return pos_counter

def compute_ttr(trees):
    tokens = []
    for t in trees:
        tokens.extend(t.leaves())

    total_tokens = len(tokens)
    types = set(tokens)
    total_types = len(types)

    if total_tokens > 0:
        return total_types / total_tokens
    else:
        return 0.0

def compute_wh_question_stats(trees):
    wh_count = 0
    for t in trees:
        for subtree in t.subtrees():
            if subtree.label().startswith('WH'):
                wh_count += 1
    return wh_count

def compute_thematic_roles(trees):
    role_counter = Counter()
    for t in trees:
        for subtree in t.subtrees():
            label = subtree.label()
            if 'AGENT' in label:
                role_counter['AGENT'] += 1
            if 'PATIENT' in label:
                role_counter['PATIENT'] += 1
            if 'EXPER' in label:
                role_counter['EXPER'] += 1
            
    return role_counter
