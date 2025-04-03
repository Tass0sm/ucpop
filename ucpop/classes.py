from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, FrozenSet

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
    
@dataclass
class Link:
    step_p: PlanStep
    condition: FNode
    step_c: PlanStep


def has_cycle(graph, start):
    """Detects cycles using an iterative DFS with an explicit stack."""
    stack = [start]  # Stack for DFS
    in_stack = set([start])  # Tracks nodes in the current DFS path
    visited = set()

    while stack:
        node = stack[-1]  # Peek the top of the stack
        if node not in visited:
            visited.add(node)

        has_unvisited_child = False
        for neighbor in graph[node]:
            if neighbor in in_stack:  # Cycle detected
                return True
            if neighbor not in visited:
                stack.append(neighbor)
                in_stack.add(neighbor)
                has_unvisited_child = True
                break  # Process one child at a time

        if not has_unvisited_child:
            stack.pop()
            in_stack.remove(node)

    return False  # No cycle detected

@dataclass
class Plan:
    steps: List[PlanStep]            # List of steps in the plan.
    ordering: List[Tuple[int, int]]  # List of (id1, id2) tuples.
    links: List[Link]                # List of causal links.
    highest_id: int = 0

    def add_link(self, step_p, condition, step_c):
        l = Link(step_p, condition, step_c)
        self.links.append(l)

    def add_step(self, step):
        if step not in self.steps:
            self.steps.append(step)
            self.highest_id = step.id

    def add_edge(self, u, v):
        if (u, v) not in self.ordering:
            self.ordering.append((u, v))

    def can_constrain(self, u, v):
        """Slow function to check if adding an edge makes the partial order
        inconsistent. TODO: change this class to be a graph anyway.
        """

        if (u, v) in self.ordering or (u == v):
            return (u, v)

        graph = defaultdict(set)

        # Build the directed graph
        for a, b in self.ordering:
            graph[a].add(b)

        # Add the new constraint
        graph[u].add(v)

        # Check for cycles using iterative DFS
        for node in self.steps:
            if has_cycle(graph, node.id):
                return None  # Cycle detected, inconsistent
        return (u, v)  # No cycle, still consistent
        
    def new_step(self, action):
        effect_conjuncts = effects_to_conjuncts(action.effects)
        return PlanStep(self.highest_id + 1, frozenset(action.preconditions), effect_conjuncts, action)
