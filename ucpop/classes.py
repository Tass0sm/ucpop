import logging
from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional, FrozenSet

from frozendict import frozendict

import unified_planning as up
from unified_planning.model import FNode, Action, Object
from unified_planning.plans import ActionInstance

from ucpop.utils import effects_to_conjuncts
from ucpop.variable import (
    Var,
    Bindings,
    DisjunctiveBindings,
    Unifier,
    DisjunctiveUnifier,
    most_general_unification
)

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class PlanStep:
    """A concrete step in a plan, which is an instance of an action."""
    id: int
    preconditions: frozenset[FNode] = field(default_factory=frozenset)
    effects: frozenset[FNode] = field(default_factory=frozenset)
    action: Optional[Action] = None

    def __repr__(self):
        # return f"PlanStep(id={self.id}, preconditions={self.preconditions}, effects={self.effects})"
        name = self.action.name if self.action else "none"
        return f"PlanStep({name}, id={self.id})"

    
@dataclass(eq=True, frozen=True)
class Link:
    step_p: PlanStep
    condition: FNode
    step_c: PlanStep


def add_edges(adj_list: frozendict[int, frozenset[int]], new_edges: dict[int, Iterable[int]]):
    new_adj_list = adj_list
    for u, new_vs in new_edges.items():
        old_vs = new_adj_list.get(u, frozenset())
        new_adj_list = new_adj_list.set(u, old_vs.union(new_vs))
    return new_adj_list


@dataclass(eq=True, frozen=True, kw_only=True)
class BasePlan:
    steps: frozendict[int, PlanStep]                # List of steps in the plan.
    adj_list: frozendict[int, frozenset[int]] # Adjacency list of step ids
    links: frozenset[Link]                    # List of causal links.
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
        return BasePlan(steps=new_steps, adj_list=new_adj_list, links=new_links, highest_id=a_add.id), a_add, new_link

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep):
        new_link = Link(a_add, q, a_need)

        new_steps = self.steps
        new_adj_list = add_edges(self.adj_list, { a_add.id: [a_need.id] })
        new_links = self.links.union({new_link})
        return BasePlan(steps=new_steps, adj_list=new_adj_list, links=new_links, highest_id=self.highest_id), new_link

    def with_new_constraint(self, first_id: int, second_id: int):
        new_adj_list = add_edges(self.adj_list, { first_id: [second_id] })
        return BasePlan(steps=self.steps, adj_list=new_adj_list, links=self.links, highest_id=self.highest_id)

    def threatens(self, step: PlanStep, link: Link):
        # TODO: add stuff for identifying if step is not already correctly ordered
        return self.possibly_between(step, link.step_p, link.step_c) and \
            ~link.condition in step.effects

    def can_constrain(self, u, v):
        """Slow function to check if adding an edge makes the partial order
        inconsistent. TODO: change this class to be a graph anyway.
        """

        if u == 0:
            return True # the start node can always be before another
        if v == -1:
            return True # the end node can always be after another
        if u == -1:
            return False # the end node can never be before another
        if v == 0:
            return False # the start node can never be after another

        # if edge already exists or its a self-loop
        if v in self.adj_list[u] or (u == v):
            return (u, v)

        # current graph with additional edge
        graph = add_edges(self.adj_list, { u: [v] })

        # contains both gray (in stack) and black nodes (popped from stack)
        visited = set()

        # Check for cycles using iterative DFS (without popping until finished
        # in order to identify cycles by checking if a node is in the stack)
        # TODO: consider changing this to just start with step 0
        for node in self.steps:

            # optimization, skip nodes that have already been DFS-ed
            if node.id in visited:
                continue

            stack = [node.id]
            gray_set = set()

            while stack:
                current = stack[-1]
                # contains both gray (in stack) and black nodes (popped from stack)
                visited |= {current}
                gray_set |= {current}

                # only pop when fully explored to keep everything in the stack
                has_white_child = False
                for child in graph.get(current, []):
                    if child in gray_set:
                        # if edge reaches back into stack, there is a cycle
                        return None
                    if child not in visited:
                        has_white_child = True
                        stack.append(child)

                if not has_white_child:
                    stack.pop()
                    gray_set -= {current}

        # No cycle, still consistent
        return (u, v)

    def can_demote(self, a_t, link):
        return self.can_constrain(a_t.id, link.step_p.id)

    def can_promote(self, a_t, link):
        return self.can_constrain(link.step_c.id, a_t.id)

    def reusable_steps(self, q: FNode, a_need: PlanStep):
        def step_could_work(step):
            return q in [e for e in step.effects] and self.possibly_before(step, a_need)

        return list(filter(step_could_work, self.steps))

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

        stack = [step.id]
        visited = set()
        while stack:
            current = stack.pop()

            if current == other.id:
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

        stack = [other.id]
        visited = set()
        while stack:
            current = stack.pop()

            if current == step.id:
                return False

            visited |= {current}

            for child in self.adj_list.get(current, []):
                if child not in visited:
                    stack.append(child)

        return True

    def to_partial_order_plan(self):
        id_to_instance_map = {}

        for step in self.steps:
            if step.id in [0, -1]:
                continue
            id_to_instance_map[step.id] = ActionInstance(step.action)

        graph = defaultdict(list)
        # Build the directed graph
        for u, vs in self.adj_list.items():
            for v in vs:
                if u in [0, -1] and v in [0, -1] or u == v:
                    continue
                elif u in [0, -1]:
                    v_inst = id_to_instance_map[v]
                    graph[v_inst] = graph[v_inst]
                elif v in [0, -1]:
                    u_inst = id_to_instance_map[u]
                    graph[u_inst] = graph[u_inst]
                else:
                    u_inst = id_to_instance_map[u]
                    v_inst = id_to_instance_map[v]
                    graph[u_inst].append(v_inst)

        return up.plans.PartialOrderPlan(graph)


@dataclass(eq=True, frozen=True, kw_only=True)
class PartialActionPlan(BasePlan):
    bindings: DisjunctiveBindings

    @staticmethod
    def _separate_preconditions(preconditions, step_id):
        def get_binding(precond):
            if precond.is_equals():
                x, y = precond.args
                x = x.parameter()
                y = y.parameter()
                return (Var(x.name, x.type, step_id), Var(y.name, y.type, step_id), True)
            elif (precond.is_not() and (~precond).is_equals()):
                x, y = (~precond).args
                x = x.parameter()
                y = y.parameter()
                return (Var(x.name, x.type, step_id), Var(y.name, y.type, step_id), False)
            else:
                return None

        binding_preconds = []
        logical_preconds = []
        for pc in preconditions:
            if (bc := get_binding(pc)):
                binding_preconds.append(bc)
            else:
                logical_preconds.append(pc)

        return binding_preconds, logical_preconds

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep, unifier: Unifier):
        new_id = self.highest_id + 1
        binding_preconds, logical_preconds = PartialActionPlan._separate_preconditions(action.preconditions, new_id)
        effect_conjuncts = effects_to_conjuncts(action.effects)
        a_add = PlanStep(new_id, frozenset(logical_preconds), effect_conjuncts, action)
        new_link = Link(a_add, q, a_need)

        new_steps = self.steps.union({a_add})
        new_adj_list = add_edges(self.adj_list, { 0: [a_add.id], a_add.id: [a_need.id, -1] })
        new_links = self.links.union({new_link})

        new_bindings = self.bindings.union(new_variables=[Var(p.name, p.type, new_id) for p in action.parameters],
                                           new_constraints=(unifier + binding_preconds))

        return (PartialActionPlan(steps=new_steps, adj_list=new_adj_list, links=new_links, bindings=new_bindings, highest_id=a_add.id),
                a_add,
                new_link,
                logical_preconds)

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep, unifier: Unifier):
        new_link = Link(a_add, q, a_need)

        new_steps = self.steps
        new_adj_list = add_edges(self.adj_list, { a_add.id: [a_need.id] })
        new_links = self.links.union({new_link})

        new_bindings = self.bindings.union(new_constraints=unifier)

        return PartialActionPlan(steps=new_steps, adj_list=new_adj_list, links=new_links, bindings=new_bindings, highest_id=self.highest_id), new_link

    def with_new_constraint(self, first_id: int, second_id: int):
        new_adj_list = add_edges(self.adj_list, { first_id: [second_id] })
        return PartialActionPlan(steps=self.steps, adj_list=new_adj_list, links=self.links, bindings=self.bindings, highest_id=self.highest_id)

    def with_new_binding(self, var: Var, obj: Object):
        new_steps = self.steps
        new_adj_list = self.adj_list
        new_links = self.links
        new_bindings = self.bindings.union(new_constraints=[(var, obj, True)])
        return PartialActionPlan(steps=new_steps, adj_list=new_adj_list, links=new_links, bindings=new_bindings, highest_id=self.highest_id)

    def reusable_steps(self, q: FNode, a_need: PlanStep):
        q_step_id = a_need.id

        def step_could_work(step):            
            unifiers = list(filter(lambda x: x is not None, map(lambda e: most_general_unification(q, q_step_id, e, step.id, self.bindings), step.effects)))
            if self.possibly_before(step, a_need) and len(unifiers) > 0:
                return step, unifiers
            else:
                return False

        reusable_steps = list(filter(None, map(step_could_work, self.steps)))
        logger.info(f"len(reusable_steps) = {len(reusable_steps)}")
        return reusable_steps


    def threatens(self, step: PlanStep, link: Link):
        if not self.possibly_between(step, link.step_p, link.step_c):
            return False

        for effect in step.effects:
            unifier = most_general_unification(~link.condition, link.step_c.id, effect, step.id, self.bindings)
            if unifier is not None and len(unifier) == 0:
                return True

        return False


@dataclass(eq=True, frozen=True, kw_only=True)
class PartialConditionalActionPlan(PartialActionPlan):
    adj_list: frozendict[int, frozenset[tuple[int, Unifier]]]            # Adjacency list of step ids
    bindings: Bindings

    # @staticmethod
    # def _separate_preconditions(preconditions, step_id):
    #     def get_binding(precond):
    #         if precond.is_equals():
    #             x, y = precond.args
    #             x = x.parameter()
    #             y = y.parameter()
    #             return (Var(x.name, x.type, step_id), Var(y.name, y.type, step_id), True)
    #         elif (precond.is_not() and (~precond).is_equals()):
    #             x, y = (~precond).args
    #             x = x.parameter()
    #             y = y.parameter()
    #             return (Var(x.name, x.type, step_id), Var(y.name, y.type, step_id), False)
    #         else:
    #             return None

    #     binding_preconds = []
    #     logical_preconds = []
    #     for pc in preconditions:
    #         if (bc := get_binding(pc)):
    #             binding_preconds.append(bc)
    #         else:
    #             logical_preconds.append(pc)

    #     return binding_preconds, logical_preconds

    @staticmethod
    def _separate_unifiers(unifiers: list[Unifier]) -> tuple[Unifier, DisjunctiveUnifier]:
        unifications = {}

        n_unifiers = len(unifiers)
        for u_i, unifier in enumerate(unifiers):
            for x, y, eq in unifier:
                assert eq, "_seperate_unifiers can only separate codesignation constraints"

                # if this unifier is binding a variable (x or y) to something record that
                if isinstance(x, Var):
                    if x in unifications:
                        unifications[x].append((u_i, y))
                    else:
                        unifications[x] = [(u_i, y)]
                if isinstance(y, Var):
                    if y in unifications:
                        unifications[y].append((u_i, x))
                    else:
                        unifications[y] = [(u_i, x)]

        constraints = []
        disjunctive_constraints = []

        # if in every unifier, a variable was ALYWAYS set to a single value add a binding constraint

        # if a variable was set to multiple values or a single value SOMETIMES
        # across unifiers, add a disjunctive binding constraint

        for var, bindings in unifications.items():
            is_binding = True
            _, c0 = bindings[0]
            cs = set()
            for i, (u_i, c) in enumerate(bindings):
                if u_i != i or c != c0:
                    is_binding = False
                cs.add(c)

            if is_binding:
                constraints.append((var, c0, True))
            else:
                disjunctive_constraints.append((var, cs))

        return constraints, disjunctive_constraints


    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep, all_unifiers: list[Unifier]):
        new_id = self.highest_id + 1
        binding_preconds, logical_preconds = PartialActionPlan._separate_preconditions(action.preconditions, new_id)
        effect_conjuncts = effects_to_conjuncts(action.effects)
        a_add = PlanStep(new_id, frozenset(logical_preconds), effect_conjuncts, action)
        new_link = Link(a_add, q, a_need)

        new_steps = self.steps | {new_id: a_add}
        new_adj_list = add_edges(self.adj_list, { 0: [(a_add.id, ())], a_add.id: [(a_need.id, ()), (-1, ())] })
        new_links = self.links.union({new_link})

        assert len(all_unifiers) > 0, "with_new_step must be called with some unifiers that allow using the step"
        new_constraints, new_disjunctive_constraints = PartialConditionalActionPlan._separate_unifiers(all_unifiers)

        new_bindings = self.bindings.union(new_variables=[Var(p.name, p.type, new_id) for p in action.parameters],
                                           new_constraints=new_constraints + binding_preconds,
                                           new_disjunctive_constraints=new_disjunctive_constraints)

        return (PartialConditionalActionPlan(steps=new_steps, adj_list=new_adj_list, links=new_links, bindings=new_bindings, highest_id=a_add.id),
                a_add,
                new_link,
                logical_preconds)

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep, all_unifiers: list[Unifier]):
        new_link = Link(a_add, q, a_need)

        new_steps = self.steps
        new_adj_list = add_edges(self.adj_list, { a_add.id: [(a_need.id, ())] })
        new_links = self.links.union({new_link})

        assert len(all_unifiers) > 0, "with_reused_step must be called with some unifiers that allow using the step"
        new_constraints, new_disjunctive_constraints = PartialConditionalActionPlan._separate_unifiers(all_unifiers)
        new_bindings = self.bindings.union(new_constraints=new_constraints,
                                           new_disjunctive_constraints=new_disjunctive_constraints)

        return PartialConditionalActionPlan(steps=new_steps, adj_list=new_adj_list, links=new_links, bindings=new_bindings, highest_id=self.highest_id), new_link

    def with_new_constraint(self, first_id: int, second_id: int, edge_conditions: frozenset[Unifier]):
        new_adj_list = add_edges(self.adj_list, { first_id: [(second_id, edge_conditions)] })
        return PartialConditionalActionPlan(steps=self.steps, adj_list=new_adj_list, links=self.links, bindings=self.bindings, highest_id=self.highest_id)

    def with_new_binding(self, var: Var, obj: Object):
        new_steps = self.steps
        new_adj_list = self.adj_list
        new_links = self.links
        new_bindings = self.bindings.union(new_constraints=[(var, obj, True)])
        return PartialConditionalActionPlan(steps=new_steps, adj_list=new_adj_list, links=new_links, bindings=new_bindings, highest_id=self.highest_id)

    def with_new_bindings(self, unifier: Unifier):
        new_steps = self.steps
        new_adj_list = self.adj_list
        new_links = self.links
        new_bindings = self.bindings.union(new_constraints=unifier)
        return PartialConditionalActionPlan(steps=new_steps, adj_list=new_adj_list, links=new_links, bindings=new_bindings, highest_id=self.highest_id)

    def reusable_steps(self, q: FNode, a_need: PlanStep):
        q_step_id = a_need.id

        def step_could_work(step):
            unifications = filter(lambda x: x is not None, map(lambda e: most_general_unification(q, q_step_id, e, step.id, self.bindings), step.effects))
            all_possible_unifiers = list(unifications)
            if self.possibly_before(step, a_need) and len(all_possible_unifiers) > 0:
                return step, all_possible_unifiers
            else:
                return False

        reusable_steps = list(filter(None, map(step_could_work, self.steps.values())))
        logger.info(f"len(reusable_steps) = {len(reusable_steps)}")
        return reusable_steps

    def threatens(self, step: PlanStep, link: Link):
        if not self.possibly_between(step, link.step_p, link.step_c):
            return None
        
        unifiers = set()
        for effect in step.effects:
            unifier = most_general_unification(~link.condition, link.step_c.id, effect, step.id, self.bindings)
            if unifier is not None:
                unifiers.add(frozenset(unifier))

        return frozenset(unifiers) or None

    def can_accommodate(self, a_t, link, requirements):
        """REQUIREMENTS is a list of conditions that, if negated,
        would ensure A_T doesn't threaten LINK."""

        unifier = []
        for x, y, eq in requirements:
            if eq:
                can_divorce, unif = self.bindings.can_divorce(x, y)
                if can_divorce:
                    unifier.extend(unif)
                else:
                    return None
            else:
                raise NotImplementedError()

        return unifier

    def can_constrain(self, u, v):

        """Slow function to check if adding an edge makes the partial order
        inconsistent. TODO: change this class to be a graph anyway.
        """

        if u == 0:
            return True # the start node can always be before another
        if v == -1:
            return True # the end node can always be after another
        if u == -1:
            return False # the end node can never be before another
        if v == 0:
            return False # the start node can never be after another

        # if edge already exists or its a self-loop
        if v in map(lambda x: x[0], self.adj_list[u]) or (u == v):
            return (u, v)

        # current graph with additional UNCONDITIONAL edge
        graph = add_edges(self.adj_list, { u: [(v, ())] })

        # contains both gray (in stack) and black nodes (popped from stack)
        visited = set()

        # Check for cycles using iterative DFS (without popping until finished
        # in order to identify cycles by checking if a node is in the stack)
        # TODO: consider changing this to just start with step 0
        for node in self.steps.values():

            # optimization, skip nodes that have already been DFS-ed
            if node.id in visited:
                continue

            stack = [node.id]
            gray_set = set()

            while stack:
                current = stack[-1]
                # contains both gray (in stack) and black nodes (popped from stack)
                visited |= {current}
                gray_set |= {current}

                # only pop when fully explored to keep everything in the stack
                has_white_child = False
                for child, _ in graph.get(current, []):
                    if child in gray_set:
                        # if edge reaches back into stack, there is a cycle
                        return None
                    if child not in visited:
                        has_white_child = True
                        stack.append(child)

                if not has_white_child:
                    stack.pop()
                    gray_set -= {current}

        # No cycle, still consistent
        return (u, v)

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

        stack = [step.id]
        visited = set()
        while stack:
            current = stack.pop()

            if current == other.id:
                return False

            visited |= {current}

            for child, _ in self.adj_list.get(current, []):
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

        stack = [other.id]
        visited = set()
        while stack:
            current = stack.pop()

            if current == step.id:
                return False

            visited |= {current}

            for child, _ in self.adj_list.get(current, []):
                if child not in visited:
                    stack.append(child)

        return True
