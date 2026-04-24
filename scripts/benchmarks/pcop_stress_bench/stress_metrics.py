from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple

import unified_planning as up


def _get_plan_nodes_and_edges(plan):
    if plan is None:
        return [], {}

    if hasattr(plan, 'get_adjacency_dicts'):
        adj = plan.get_adjacency_dicts
        return list(adj.keys()), adj

    if hasattr(plan, 'get_adjacency_list'):
        adj_list = plan.get_adjacency_list
        adj = {node: {succ: {} for succ in succs} for node, succs in adj_list.items()}
        return list(adj.keys()), adj

    if hasattr(plan, 'actions'):
        actions = list(plan.actions)
        return actions, {a: {} for a in actions}

    return [], {}


def _count_transitively_comparable_pairs(adj: Dict[object, Dict[object, dict]]) -> int:
    nodes = list(adj.keys())
    if not nodes:
        return 0

    comparable = 0
    for start in nodes:
        seen = set()
        q = deque(adj.get(start, {}).keys())
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            comparable += 1
            for nxt in adj.get(cur, {}).keys():
                if nxt not in seen:
                    q.append(nxt)
    return comparable


def _safe_object_name(fnode) -> Tuple[str | None, str | None]:
    try:
        if hasattr(fnode, 'is_object_exp') and fnode.is_object_exp():
            obj = fnode.object()
            type_name = getattr(obj.type, 'name', str(obj.type))
            return str(obj), str(type_name)
    except Exception:
        pass
    try:
        type_name = getattr(fnode.type, 'name', str(fnode.type))
        return str(fnode), str(type_name)
    except Exception:
        return str(fnode), None


def _action_instances(plan) -> Iterable[object]:
    nodes, _ = _get_plan_nodes_and_edges(plan)
    return nodes


def extract_plan_metrics(plan, *, sync_actions=None, resource_actions=None) -> dict:
    sync_actions = sync_actions or set()
    resource_actions = resource_actions or set()

    nodes, adj = _get_plan_nodes_and_edges(plan)
    num_actions = len(nodes)
    direct_edges = sum(len(v) for v in adj.values())
    total_pairs = num_actions * (num_actions - 1) // 2
    comparable_pairs = _count_transitively_comparable_pairs(adj)
    unordered_pairs = max(total_pairs - comparable_pairs, 0)
    flexibility_ratio = round(unordered_pairs / total_pairs, 6) if total_pairs else 0.0

    unique_agents = set()
    sync_count = 0
    resource_count = 0
    action_name_hist = {}

    for ai in _action_instances(plan):
        action_name = getattr(getattr(ai, 'action', None), 'name', '')
        action_name_hist[action_name] = action_name_hist.get(action_name, 0) + 1
        if action_name in sync_actions:
            sync_count += 1
        if action_name in resource_actions:
            resource_count += 1

        params = getattr(ai, 'actual_parameters', tuple())
        for p in params:
            name, type_name = _safe_object_name(p)
            if type_name == 'Agent' or type_name == 'Rover':
                unique_agents.add(name)

    relevant_bindings = getattr(plan, '_relevant_variable_bindings', {})
    num_binding_vars = len(relevant_bindings)
    total_possible_bindings = 0
    total_invalid_bindings = 0
    for value in relevant_bindings.values():
        try:
            possible, invalid = value
            total_possible_bindings += len(possible)
            total_invalid_bindings += len(invalid)
        except Exception:
            pass

    return {
        'action_count': num_actions,
        'direct_orderings': direct_edges,
        'comparable_pairs': comparable_pairs,
        'unordered_pairs': unordered_pairs,
        'flexibility_ratio': flexibility_ratio,
        'unique_agent_count': len(unique_agents),
        'sync_action_count': sync_count,
        'resource_action_count': resource_count,
        'binding_var_count': num_binding_vars,
        'possible_binding_count': total_possible_bindings,
        'invalid_binding_count': total_invalid_bindings,
        'action_histogram': ';'.join(f'{k}:{v}' for k, v in sorted(action_name_hist.items())),
    }


def is_solved(status) -> bool:
    return status in {
        up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING,
        up.engines.PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    }


def plan_kind(plan) -> str:
    return '' if plan is None else type(plan).__name__
