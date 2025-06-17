"""Implementation of POP supporting action schemata based on
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""

import logging
from enum import Enum
from itertools import chain
from typing import List, Dict, Tuple, Collection, FrozenSet, Any
from dataclasses import dataclass

from frozendict import frozendict
from unified_planning.model import FNode, Problem, Action, Effect, Object

from ucpop.search import best_first_search
from ucpop.classes import PlanStep, Link, PartialActionPlan as Plan
from ucpop.utils import initial_values_to_conjuncts
from ucpop.variable import Var, Bindings, Unifier, most_general_unification

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class POP2SearchNode:
    plan: Plan
    agenda: FrozenSet[FNode]
    threats: FrozenSet[Tuple[PlanStep, Link]]

    # Nota Bene, threats are considered part of the search node so that they are
    # lazily resolved during search, as opposed to eagerly resolved during the
    # generation of daughter plans from agenda items

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep, unifier: Unifier):
        new_plan, a_add, new_link, logical_preconds = self.plan.with_new_step(action, q, a_need, unifier)

        # add logical preconditions to agenda
        new_agenda = self.agenda - {(q, a_need)} | set([(pc, a_add) for pc in logical_preconds])

        # test all steps in new plan to identify which threaten the new link
        threats_to_new_link = []
        for step in new_plan.steps:
            if new_plan.threatens(step, new_link):
                threats_to_new_link.append((step, new_link))

        # test all links in new plan to identify which are threatened by the new step
        threats_from_new_step = []
        for link in new_plan.links:
            if new_plan.threatens(a_add, link):
                threats_from_new_step.append((a_add, link))

        added_threats = threats_to_new_link + threats_from_new_step
        new_threats = self.threats.union(added_threats)

        logger.info(f"Making plan using new action {action.name} with id {a_add.id} and unifier {unifier}")
        return POP2SearchNode(new_plan, new_agenda, new_threats)

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep, unifier: Unifier):
        new_plan, new_link = self.plan.with_reused_step(a_add, q, a_need, unifier)
        new_agenda = self.agenda - {(q, a_need)}

        # test all steps in new plan to identify which threaten the new link
        threats_to_new_link = []
        for step in new_plan.steps:
            if new_plan.threatens(step, new_link):
                threats_to_new_link.append((step, new_link))

        new_threats = self.threats.union(threats_to_new_link)

        logger.info(f"Making plan reusing step: {a_add.action.name if a_add.action else None} with id {a_add.id} and unifier {unifier}")
        return POP2SearchNode(new_plan, new_agenda, new_threats)

    def with_new_binding(self, var: Var, obj: Object):
        new_plan = self.plan.with_new_binding(var, obj)
        new_agenda = self.agenda

        # there can be a faster way to test for new threats, but this is the
        # simplest version for now.
        new_threats_from_binding = []
        for step in new_plan.steps:
            for link in new_plan.links:
                if new_plan.threatens(step, link):
                    new_threats_from_binding.append((step, link))

        new_threats = self.threats.union(new_threats_from_binding)
        return POP2SearchNode(new_plan, new_agenda, new_threats)

    def with_new_constraint(self, first_id: int, second_id: int):
        new_plan = self.plan.with_new_constraint(first_id, second_id)
        return POP2SearchNode(new_plan, self.agenda, self.threats)

    def __le__(self, other):
        """Compare the size of self.plan to other.plan"""
        return (len(self.plan.steps) + len(self.agenda) + len(self.threats)) <= \
            (len(other.plan.steps) + len(other.agenda) + len(other.threats))

    def __lt__(self, other):
        """Compare the size of self.plan to other.plan"""
        return (len(self.plan.steps) + len(self.agenda) + len(self.threats)) < \
            (len(other.plan.steps) + len(other.agenda) + len(other.threats))


class POP2FlawType(Enum):
    THREAT = 0
    OPENCOND = 1
    UNBOUND_VAR = 2


class POP2:

    def __init__(self, problem: Problem):
        self.problem = problem

    def _create_initial_node(self) -> POP2SearchNode:
        # Plan
        start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(self.problem.initial_values))
        end_step = PlanStep(id=-1, preconditions=frozenset(self.problem.goals))

        plan = Plan(steps=frozenset([start_step, end_step]),
                    adj_list=frozendict({0: frozenset({-1})}),
                    bindings=Bindings.empty(),
                    links=frozenset())

        # Agenda
        agenda = frozenset([(q, end_step) for q in end_step.preconditions])

        # Threats
        threats = frozenset([])

        return POP2SearchNode(plan, agenda, threats)

    def _get_flaw(self, node: POP2SearchNode) -> Tuple[Any, POP2FlawType]:
        """This corresponds to the goal selection step in the POP
        non-deterministic pseudocode"""
        if node.threats:
            return next(iter(node.threats)), POP2FlawType.THREAT
        elif node.agenda:
            return next(iter(node.agenda)), POP2FlawType.OPENCOND
        elif node.plan.bindings.unbound:
            return next(iter(node.plan.bindings.unbound)), POP2FlawType.UNBOUND_VAR
        else:
            return None, None

    def _get_daughter_nodes_for_threat(self, node: POP2SearchNode, flaw):
        """This corresponds to the causal link protection step in the POP
        non-deterministic pseudocode"""
        a_t, link = flaw
        demotion_c = node.plan.can_demote(a_t, link)
        promotion_c = node.plan.can_promote(a_t, link)
        if demotion_c and promotion_c:
            return [node.with_new_constraint(*demotion_c),
                    node.with_new_constraint(*promotion_c)]
        elif demotion_c:
            return [node.with_new_constraint(*demotion_c)]
        elif promotion_c:
            return [node.with_new_constraint(*promotion_c)]
        else:
            # Couldn't promote or demote to resolve threat so no daughter plans
            return []

    def _get_daughter_nodes_for_opencond(self, node: POP2SearchNode, flaw):
        """This corresponds to the action selection step in the POP
        non-deterministic pseudocode"""
        q, a_need = flaw
        q_step_id = a_need.id

        def action_could_work(action):
            # iterate over effects to find one that unifies with goal
            unifications = filter(lambda x: x is not None,
                                  map(lambda e: most_general_unification(q, q_step_id, e,
                                                                         node.plan.highest_id + 1,
                                                                         node.plan.bindings),
                                      action.effects))

            # return that unification if found
            return list(unifications)

        def plans_with_new_step_from(action, unifiers):
            return [node.with_new_step(action, q, a_need, unifier) for unifier in unifiers]

        # possible plans when a_add is a newly instantiated step
        new_step_plans = list(chain.from_iterable(
            [plans_with_new_step_from(action, unifiers)
             for action in self.problem.actions
             if len(unifiers := action_could_work(action)) > 0]))

        def plans_with_reused_step_from(step, unifiers):
            return [node.with_reused_step(step, q, a_need, unifier) for unifier in unifiers]

        logger.info(f"Getting reusable steps for ({q}, {q_step_id}) under {node.plan.bindings}")

        # possible plans when a_add is a reused step
        reused_step_plans = list(chain.from_iterable(
            [plans_with_reused_step_from(step, unifiers)
             for step, unifiers in node.plan.reusable_steps(q, a_need)]))

        # all possible plans derived from all possible choices of a_add
        daughter_plans = new_step_plans + reused_step_plans
        return daughter_plans


    def _get_daughter_nodes_for_unbound_var(self, node: POP2SearchNode, flaw):
        """Generate nodes resolving unbound variable."""
        unbound_var = flaw
        objects = self.problem.objects(unbound_var.type)

        def can_unify(o):
            r, _ = node.plan.bindings.can_unify(o, unbound_var)
            return r

        objects = filter(can_unify, objects)

        return [node.with_new_binding(unbound_var, o) for o in objects]

    def _get_daughter_nodes_for_flaw(self, node: POP2SearchNode, flaw_type: POP2FlawType, flaw):
        if flaw_type == POP2FlawType.THREAT:
            return self._get_daughter_nodes_for_threat(node, flaw)
        elif flaw_type == POP2FlawType.OPENCOND:
            return self._get_daughter_nodes_for_opencond(node, flaw)
        elif flaw_type == POP2FlawType.UNBOUND_VAR:
            return self._get_daughter_nodes_for_unbound_var(node, flaw)
        else:
            return []

    def execute(self, search_limit=2000):
        """Run the search in plan-space.
        """

        # step 1
        def pop_goal_p(node):
            return len(node.agenda) == 0 and len(node.threats) == 0 and len(node.plan.bindings.unbound) == 0

        def pop_daughters_fn(node):
            # step 2
            flaw, flaw_type = self._get_flaw(node)
            logger.info(f"Addressing {flaw_type} {flaw}")
            # step 3
            daughter_nodes = self._get_daughter_nodes_for_flaw(node, flaw_type, flaw)
            # logger.info(f"Threats: {current.threats}")
            logger.info(f"Num Daughters = {len(daughter_nodes)}")
            return daughter_nodes

        def pop_rank_fn(node):
            return len(node.plan.steps) + len(node.agenda) + len(node.threats)

        node = self._create_initial_node()
        goal_node = best_first_search(node,
                                      pop_daughters_fn,
                                      pop_goal_p,
                                      pop_rank_fn,
                                      search_limit)

        return goal_node.plan if goal_node else None
