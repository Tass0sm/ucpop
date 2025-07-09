"""Implementation of POP based on
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""

import logging
from enum import Enum
from typing import Tuple, FrozenSet, Any
from functools import reduce
from dataclasses import dataclass

from frozendict import frozendict
from unified_planning.model import FNode, Problem, Action
from unified_planning.shortcuts import Exists, Forall

from ucpop.search import best_first_search
from ucpop.classes import PlanStep, Link, BasePlan as Plan
from ucpop.utils import initial_values_to_conjuncts

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class POPSearchNode:
    plan: Plan
    agenda: FrozenSet[FNode]
    threats: FrozenSet[Tuple[PlanStep, Link]]

    # Nota Bene, threats are considered part of the search node so that they are
    # lazily resolved during search, as opposed to eagerly resolved during the
    # generation of daughter plans from agenda items

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep):
        new_plan, a_add, new_link = self.plan.with_new_step(action, q, a_need)
        new_agenda = self.agenda - {(q, a_need)} | set([(pc, a_add) for pc in a_add.preconditions])

        # logger.info(f"Making node with new step: {a_add}\n and new link: {new_link}")

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

        logger.info(f"Making plan using new action {action.name} with id {a_add.id}")
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

        logger.info(f"Making plan reusing step: {a_add.action.name if a_add.action else None} with id {a_add.id}")
        return POPSearchNode(new_plan, new_agenda, new_threats)

    def with_new_constraint(self, first_id: int, second_id: int):
        new_plan = self.plan.with_new_constraint(first_id, second_id)
        return POPSearchNode(new_plan, self.agenda, self.threats)

    def with_new_agenda(self, new_agenda):
        return POPSearchNode(self.plan, new_agenda, self.threats)

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
    UNIVERSAL_GOAL = 2
    CONJUNCTION_GOAL = 3

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

    def _compute_universal_base(self, q: FNode) -> FNode:
        # first, take out of outer not
        if q.is_not() and (~q).is_exists():
            variable = (~q).variables()[0]
            clause = ~(~q).arg(0)
            q = Forall(clause, variable)

        if q.is_forall():
            clauses = [clause.substitute({variable: obj}) for obj in self.problem.objects(variable.type)]
            q = reduce(lambda x, y: x & y, clauses).simplify()

        return q

    def _get_flaw(self, node: POPSearchNode) -> Tuple[Any, POPFlawType]:
        """This corresponds to the goal selection step in the POP
        non-deterministic pseudocode"""
        if node.threats:
            return next(iter(node.threats)), POPFlawType.THREAT
        elif node.agenda:
            q, a_c = next(iter(node.agenda))

            if (~q).is_exists() or q.is_exists() or (~q).is_forall() or q.is_forall():
                return (q, a_c), POPFlawType.UNIVERSAL_GOAL
            elif q.is_and():
                return (q, a_c), POPFlawType.CONJUNCTION_GOAL
            else:
                return (q, a_c), POPFlawType.OPENCOND
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

    def _get_daughter_nodes_for_universal_goal(self, node: POPSearchNode, flaw):
        """This function is a workaround for the complicated step 2 in UCPOP,
        which resolves a universal goal from the agenda."""
        q, a_need = flaw
        q_base = self._compute_universal_base(q)
        new_agenda = node.agenda - {(q, a_need)} | {(q_base, a_need)}
        return [node.with_new_agenda(new_agenda)]

    def _get_daughter_nodes_for_conjunction_goal(self, node: POPSearchNode, flaw):
        """This function is a workaround for the complicated step 2 in UCPOP,
        which resolves a conjunction goal from the agenda."""
        q, a_need = flaw
        new_agenda = node.agenda - {(q, a_need)} | set([(arg, a_need) for arg in q.args])
        return [node.with_new_agenda(new_agenda)]

    def _get_daughter_nodes_for_flaw(self, node: POPSearchNode, flaw_type: POPFlawType, flaw):
        if flaw_type == POPFlawType.THREAT:
            return self._get_daughter_nodes_for_threat(node, flaw)
        elif flaw_type == POPFlawType.OPENCOND:
            return self._get_daughter_nodes_for_opencond(node, flaw)
        elif flaw_type == POPFlawType.UNIVERSAL_GOAL:
            return self._get_daughter_nodes_for_universal_goal(node, flaw)
        elif flaw_type == POPFlawType.CONJUNCTION_GOAL:
            return self._get_daughter_nodes_for_conjunction_goal(node, flaw)
        else:
            return None

    def execute(self, search_limit=40000):
        """Run the search in plan-space.
        """

        # step 1
        def pop_goal_p(node):
            return len(node.agenda) == 0 and len(node.threats) == 0

        def pop_daughters_fn(node):
            # step 2
            flaw, flaw_type = self._get_flaw(node)
            logger.info(f"Addressing {flaw_type} {flaw}")
            # step 3
            daughter_nodes = self._get_daughter_nodes_for_flaw(node, flaw_type, flaw)
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
