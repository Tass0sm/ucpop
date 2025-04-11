from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, FrozenSet

from frozendict import frozendict
from unified_planning.model import FNode, Action

from ucpop.utils import effects_to_conjuncts


@dataclass(eq=True, frozen=True)
class PlanStep:
    """A concrete step in a plan, which is an instance of an action."""
    id: int
    preconditions: FrozenSet[FNode] = field(default_factory=frozenset)
    effects: FrozenSet[FNode] = field(default_factory=frozenset)
    action: Optional[Action] = 1,

    def __repr__(self):
        return f"PlanStep(id={self.id}, preconditions={self.preconditions}, effects={self.effects})"
    
@dataclass(eq=True, frozen=True)
class Link:
    step_p: PlanStep
    condition: FNode
    step_c: PlanStep


# def has_cycle(graph, start):
#     """Detects cycles using an iterative DFS with an explicit stack."""
#     stack = [start]  # Stack for DFS
#     in_stack = set([start])  # Tracks nodes in the current DFS path
#     visited = set()

#     while stack:
#         node = stack[-1]  # Peek the top of the stack
#         if node not in visited:
#             visited.add(node)

#         has_unvisited_child = False
#         for neighbor in graph[node]:
#             if neighbor in in_stack:  # Cycle detected
#                 return True
#             if neighbor not in visited:
#                 stack.append(neighbor)
#                 in_stack.add(neighbor)
#                 has_unvisited_child = True
#                 break  # Process one child at a time

#         if not has_unvisited_child:
#             stack.pop()
#             in_stack.remove(node)

#     return False  # No cycle detected


def add_edges(adj_list: frozendict[int, FrozenSet[int]], new_edges: dict[int, Iterable[int]]):
    new_adj_list = adj_list
    for u, new_vs in new_edges.items():
        old_vs = new_adj_list.get(u, frozenset())
        new_adj_list = new_adj_list.set(u, old_vs.union(new_vs))
    return new_adj_list


@dataclass(eq=True, frozen=True)
class Plan:
    steps: FrozenSet[PlanStep]                # List of steps in the plan.
    adj_list: frozendict[int, int]            # Adjacency list of step ids
    links: FrozenSet[Link]                    # List of causal links.
    highest_id: int = 0

    # transitive_closure: FrozenDict[int, int]  # Transitive closure of this graph.

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep):
        effect_conjuncts = effects_to_conjuncts(action.effects)
        a_add = PlanStep(self.highest_id + 1, frozenset(action.preconditions), effect_conjuncts, action)
        new_link = Link(a_add, q, a_need)

        new_steps = self.steps.union({a_add})
        new_adj_list = add_edges(self.adj_list, { 0: [a_add.id],
                                                  a_add.id: [a_need.id, -1] })
        new_links = self.links.union({new_link})
        return Plan(new_steps, new_adj_list, new_links, a_add.id), a_add, new_link

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep):
        new_link = Link(a_add, q, a_need)

        new_steps = self.steps
        new_adj_list = add_edges(self.adj_list, { a_add.id: [a_need.id] })
        new_links = self.links.union({new_link})
        return Plan(new_steps, new_adj_list, new_links, self.highest_id), new_link

    def threatens(self, step: PlanStep, link: Link):
        # TODO: add stuff for identifying if step is not already correctly ordered
        return self.possibly_between(step, link.step_p, link.step_c) and \
            ~link.condition in step.effects

    def possibly_between(self, step: PlanStep, step_a: PlanStep, step_b: PlanStep):
        return self.possibly_after(step, step_a) and self.possibly_before(step, step_b)

    # TODO: Implement the two following functions with a maintained transitive
    # closure graph.
    
    def possibly_after(self, step: PlanStep, other: PlanStep):
        """Current slow implementation: DFS from step to see if other is
        reachable. If other is reachable, step is not possibly after other
        (False). Otherwise true.

        """
        if other == step:
            return False # strict partial order?
        if step.id == 0:
            return False # the start node is never after anything else
        if step.id == -1:
            return True # the end node is always after anything else
        if other.id == 0:
            return True # everything is after the start node
        if other.id == -1:
            return False # nothing can be after the end node
        if step.id not in self.adj_list:
            return False # if step is not in the graph anywhere, consider it to
                         # not be part of the partial order

        stack = [step]
        visited = set()
        while stack:
            current = stack.pop()

            if current == other:
                return False

            visited |= {current}

            for child in self.adj_list.get(current, []):
                if child not in visited:
                    stack.append(child)

        return True

    def possibly_before(self, step: PlanStep, other: PlanStep):
        """Current slow implementation: DFS from other to see if step is
        reachable. If its reachable, then step is not possibly before other
        (false). Otherwise true.

        """
        if step == other:
            return False # strict partial order?
        if step.id == 0:
            return True # the start node is always before anything else
        if step.id == -1:
            return False # the end node can't be possibly before anything else
        if other.id == 0:
            return False # nothing can be before the start node
        if other.id == -1:
            return True # everything is before the end node
        if other.id not in self.adj_list:
            return False # if step is not in the graph anywhere, consider it to
                         # not be part of the partial order

        stack = [other]
        visited = set()
        while stack:
            current = stack.pop()

            if current == step:
                return False

            visited |= {current}

            for child in self.adj_list.get(current, []):
                if child not in visited:
                    stack.append(child)

        return True

    # def add_link(self, step_p, condition, step_c):
    #     l = Link(step_p, condition, step_c)
    #     self.links.append(l)

    # def add_step(self, step):
    #     if step not in self.steps:
    #         self.steps.append(step)
    #         self.highest_id = step.id

    # def add_edge(self, u, v):
    #     if (u, v) not in self.ordering:
    #         self.ordering.append((u, v))

    # def can_constrain(self, u, v):
    #     """Slow function to check if adding an edge makes the partial order
    #     inconsistent. TODO: change this class to be a graph anyway.
    #     """

    #     if (u, v) in self.ordering or (u == v):
    #         return (u, v)

    #     graph = defaultdict(set)

    #     # Build the directed graph
    #     for a, b in self.ordering:
    #         graph[a].add(b)

    #     # Add the new constraint
    #     graph[u].add(v)

    #     # Check for cycles using iterative DFS
    #     for node in self.steps:
    #         if has_cycle(graph, node.id):
    #             return None  # Cycle detected, inconsistent
    #     return (u, v)  # No cycle, still consistent

    # def new_step(self, action):
    #     effect_conjuncts = effects_to_conjuncts(action.effects)
    #     return PlanStep(self.highest_id + 1, frozenset(action.preconditions), effect_conjuncts, action)


# @dataclass
# class PlanFlaws:
#     agenda: List[PlanStep]            # List of steps in the plan.
#     ordering: List[Tuple[int, int]]  # List of (id1, id2) tuples.
#     links: List[Link]                # List of causal links.
#     highest_id: int = 0
