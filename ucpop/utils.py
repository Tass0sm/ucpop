from typing import List, Dict
from unified_planning.model import FNode, Effect


def effects_to_conjuncts(effects: List[Effect]):
    conjuncts = []
    for e in effects:
        if e.value.is_true():
            conjuncts.append(e.fluent)
        else:
            conjuncts.append(~e.fluent)

    return frozenset(conjuncts)


def initial_values_to_conjuncts(initial_values: Dict[FNode, FNode]):
    conjuncts = []
    for f, v in initial_values.items():
        if v.is_true():
            conjuncts.append(f)
        else:
            conjuncts.append(~f)

    return frozenset(conjuncts)
