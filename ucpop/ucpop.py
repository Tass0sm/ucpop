"""Implementation of POP based on
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""

import random
from enum import Enum
from typing import List, Dict, Tuple, FrozenSet, Any
from dataclasses import dataclass

from frozendict import frozendict
from unified_planning.model import FNode, Problem, Action, Effect

from ucpop.search import best_first_search
from ucpop.classes import PlanStep, Link, Plan
from ucpop.utils import initial_values_to_conjuncts



@dataclass(eq=True, frozen=True)
class UCPOPSearchNode:
    plan: Plan
    agenda: FrozenSet[FNode]
    threats: FrozenSet[Tuple[PlanStep, Link]]

    # Nota Bene, threats are considered part of the search node so that they are
    # lazily resolved during search, as opposed to eagerly resolved during the
    # generation of daughter plans from agenda items

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep):
        new_plan, a_add, new_link = self.plan.with_new_step(action, q, a_need)
        new_agenda = self.agenda - {(q, a_need)} | set([(pc, a_add) for pc in a_add.preconditions])

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
        return POPSearchNode(new_plan, new_agenda, new_threats)

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep):
        new_plan, new_link = self.plan.with_reused_step(a_add, q, a_need)
        new_agenda = self.agenda - {(q, a_need)}

        # test all steps in new plan to identify which threaten the new link
        threats_to_new_link = []
        for step in new_plan.steps:
            if new_plan.threatens(step, new_link):
                threats_to_new_link.append((step, new_link))

        new_threats = self.threats.union(threats_to_new_link)
        return POPSearchNode(new_plan, new_agenda, new_threats)

    def with_new_constraint(self, first_id: int, second_id: int):
        new_plan = self.plan.with_new_constraint(first_id, second_id)
        return POPSearchNode(new_plan, self.agenda, self.threats)

    def __le__(self, other):
        """Compare the size of self.plan to other.plan"""
        return (len(self.plan.steps) + len(self.agenda) + len(self.threats)) <= \
            (len(other.plan.steps) + len(other.agenda) + len(other.threats))

    def __lt__(self, other):
        """Compare the size of self.plan to other.plan"""
        return (len(self.plan.steps) + len(self.agenda) + len(self.threats)) < \
            (len(other.plan.steps) + len(other.agenda) + len(other.threats))


class POPFlawType(Enum):
    THREAT = 0
    OPENCOND = 1


class POP:

    def __init__(self, problem: Problem):
        self.problem = problem

    def _create_initial_node(self) -> POPSearchNode:
        # Plan
        start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(self.problem.initial_values))
        end_step = PlanStep(id=-1, preconditions=frozenset(self.problem.goals))
        plan = Plan(steps=frozenset([start_step, end_step]),
                    adj_list=frozendict({0: frozenset({-1})}),
                    links=frozenset())

        # Agenda
        agenda = frozenset([(q, end_step) for q in end_step.preconditions])

        # Threats
        threats = frozenset([])

        return POPSearchNode(plan, agenda, threats)

    def _get_flaw(self, node: POPSearchNode) -> Tuple[Any, POPFlawType]:
        """This corresponds to the goal selection step in the POP
        non-deterministic pseudocode"""
        if node.threats:
            return next(iter(node.threats)), POPFlawType.THREAT
        elif node.agenda:
            return next(iter(node.agenda)), POPFlawType.OPENCOND
        else:
            return None, None

    def _get_daughter_nodes_for_threat(self, node: POPSearchNode, flaw):
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

    def _get_daughter_nodes_for_opencond(self, node: POPSearchNode, flaw):
        """This corresponds to the action selection step in the POP
        non-deterministic pseudocode"""
        q, a_need = flaw

        def action_could_work(action):
            return q in [e.fluent for e in action.effects if e.value.is_true()]

        def plan_with_new_step_from(action):
            return node.with_new_step(action, q, a_need)

        # possible plans when a_add is a newly instantiated step
        new_step_plans = [plan_with_new_step_from(action) for action in self.problem.actions if action_could_work(action)]

        def plan_with_reused_step_from(step):
            return node.with_reused_step(step, q, a_need)

        # possible plans when a_add is a reused step
        reused_step_plans = [plan_with_reused_step_from(step) for step in node.plan.reusable_steps(q, a_need)]

        # all possible plans derived from all possible choices of a_add
        daughter_plans = new_step_plans + reused_step_plans
        return daughter_plans


    def _get_daughter_nodes_for_flaw(self, node: POPSearchNode, flaw_type: POPFlawType, flaw):
        if flaw_type == POPFlawType.THREAT:
            return self._get_daughter_nodes_for_threat(node, flaw)
        elif flaw_type == POPFlawType.OPENCOND:
            return self._get_daughter_nodes_for_opencond(node, flaw)
        else:
            return None

    def execute(self, search_limit=2000):
        """Run the search in plan-space.
        """

        # step 1
        def pop_goal_p(node):
            return len(node.agenda) == 0 and len(node.threats) == 0

        def pop_daughters_fn(node):
            # step 2
            flaw, flaw_type = self._get_flaw(node)
            # step 3
            daughter_nodes = self._get_daughter_nodes_for_flaw(node, flaw_type, flaw)
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
