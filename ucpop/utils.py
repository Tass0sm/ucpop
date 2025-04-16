from typing import List, Dict
from unified_planning.model import FNode, Effect


def effect_to_conjunct(e: Effect):
    return e.fluent if e.value.is_true() else ~e.fluent


def effects_to_conjuncts(effects: List[Effect]):
    return frozenset(map(effect_to_conjunct, effects))


def initial_values_to_conjuncts(initial_values: Dict[FNode, FNode]):
    conjuncts = []
    for f, v in initial_values.items():
        if v.is_true():
            conjuncts.append(f)
        else:
            conjuncts.append(~f)

    return frozenset(conjuncts)
