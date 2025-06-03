"""Implementation of Partial Conditional Ordering Planning
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""

import random
import logging
from enum import Enum
from typing import List, Dict, Tuple, Collection, FrozenSet, Any
from dataclasses import dataclass

from frozendict import frozendict
from unified_planning.model import FNode, Problem, Action, Effect, Object

from ucpop.search import best_first_search
from ucpop.classes import PlanStep, Link, PartialConditionalActionPlan as Plan
from ucpop.utils import initial_values_to_conjuncts
from ucpop.variable import Var, DisjunctiveBindings as Bindings, Unifier, most_general_unification

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class PCOPSearchNode:
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
            if (threat_conditions := new_plan.threatens(step, new_link)) is not None:
                threats_to_new_link.append((step, new_link, threat_conditions))

        # test all links in new plan to identify which are threatened by the new step
        threats_from_new_step = []
        for link in new_plan.links:
            if (threat_conditions := new_plan.threatens(a_add, link)) is not None:
                threats_from_new_step.append((a_add, link, threat_conditions))

        added_threats = threats_to_new_link + threats_from_new_step
        new_threats = self.threats.union(added_threats)

        logger.info(f"Making plan using new action {action.name} with id {a_add.id}")
        return PCOPSearchNode(new_plan, new_agenda, new_threats)

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep, all_unifiers: list[Unifier]):
        new_plan, new_link = self.plan.with_reused_step(a_add, q, a_need, all_unifiers)
        new_agenda = self.agenda - {(q, a_need)}

        # test all steps in new plan to identify which threaten the new link
        threats_to_new_link = []
        for step in new_plan.steps:
            if (threat_conditions := new_plan.threatens(step, new_link)) is not None:
                threats_to_new_link.append((step, new_link, threat_conditions))

        new_threats = self.threats.union(threats_to_new_link)

        logger.info(f"Making plan reusing step: {a_add.action.name if a_add.action else None} with id {a_add.id}")
        return PCOPSearchNode(new_plan, new_agenda, new_threats)

    def with_new_constraint(
            self,
            first_id: int,
            second_id: int,
            edge_conditions: frozenset[Unifier],
            addressed_threat
    ):
        new_plan = self.plan.with_new_constraint(first_id, second_id, edge_conditions)
        new_threats = self.threats - {addressed_threat}
        return PCOPSearchNode(new_plan, self.agenda, new_threats)

    def __le__(self, other):
        """Compare the size of self.plan to other.plan"""
        return (len(self.plan.steps) + len(self.agenda) + len(self.threats)) <= \
            (len(other.plan.steps) + len(other.agenda) + len(other.threats))

    def __lt__(self, other):
        """Compare the size of self.plan to other.plan"""
        return (len(self.plan.steps) + len(self.agenda) + len(self.threats)) < \
            (len(other.plan.steps) + len(other.agenda) + len(other.threats))


class PCOPFlawType(Enum):
    THREAT = 0
    OPENCOND = 1


class PCOP:

    def __init__(self, problem: Problem):
        self.problem = problem

    def _create_initial_node(self) -> PCOPSearchNode:
        # Plan
        start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(self.problem.initial_values))
        end_step = PlanStep(id=-1, preconditions=frozenset(self.problem.goals))

        plan = Plan(steps=frozenset([start_step, end_step]),
                    adj_list=frozendict({0: frozenset({(-1, ())})}),
                    bindings=Bindings.empty(),
                    links=frozenset())

        # Agenda
        agenda = frozenset([(q, end_step) for q in end_step.preconditions])

        # Threats
        threats = frozenset([])

        return PCOPSearchNode(plan, agenda, threats)

    def _get_flaw(self, node: PCOPSearchNode) -> Tuple[Any, PCOPFlawType]:
        """This corresponds to the goal selection step in the POP
        non-deterministic pseudocode"""
        if node.threats:
            return next(iter(node.threats)), PCOPFlawType.THREAT
        elif node.agenda:
            return next(iter(node.agenda)), PCOPFlawType.OPENCOND
        else:
            return None, None

    def _get_daughter_nodes_for_threat(self, node: PCOPSearchNode, flaw):
        """This corresponds to the causal link protection step in the POP
        non-deterministic pseudocode"""
        a_t, link, threat_conditions = flaw
        demotion_c = node.plan.can_demote(a_t, link)
        promotion_c = node.plan.can_promote(a_t, link)
        if demotion_c and promotion_c:
            return [node.with_new_constraint(*demotion_c, threat_conditions, flaw),
                    node.with_new_constraint(*promotion_c, threat_conditions, flaw)]
        elif demotion_c:
            return [node.with_new_constraint(*demotion_c, threat_conditions, flaw)]
        elif promotion_c:
            return [node.with_new_constraint(*promotion_c, threat_conditions, flaw)]
        else:
            # Couldn't promote or demote to resolve threat so no daughter plans
            return []

    def _get_daughter_nodes_for_opencond(self, node: PCOPSearchNode, flaw):
        """This corresponds to the action selection step in the POP
        non-deterministic pseudocode"""
        q, a_need = flaw
        q_step_id = a_need.id

        def action_could_work(action):
            # iterate over effects to find one that unifies with goal
            unifications = filter(lambda x: x is not None, map(lambda e: most_general_unification(q, q_step_id, e, node.plan.highest_id + 1, node.plan.bindings), action.effects))
            all_possible_unifiers = list(unifications)
            return all_possible_unifiers

        def plan_with_new_step_from(action, all_unifiers):
            return node.with_new_step(action, q, a_need, all_unifiers)

        # possible plans when a_add is a newly instantiated step
        new_step_plans = [plan_with_new_step_from(action, all_unifiers) for action in self.problem.actions if len(all_unifiers := action_could_work(action)) > 0]

        def plan_with_reused_step_from(step, all_unifiers):
            return node.with_reused_step(step, q, a_need, all_unifiers)

        logger.info(f"Getting reusable steps for ({q}, {q_step_id}) under {node.plan.bindings}")

        # possible plans when a_add is a reused step
        reused_step_plans = [plan_with_reused_step_from(step, all_unifiers) for step, all_unifiers in node.plan.reusable_steps(q, a_need)]

        # all possible plans derived from all possible choices of a_add
        daughter_plans = new_step_plans + reused_step_plans
        return daughter_plans


    def _get_daughter_nodes_for_flaw(self, node: PCOPSearchNode, flaw_type: PCOPFlawType, flaw):
        if flaw_type == PCOPFlawType.THREAT:
            return self._get_daughter_nodes_for_threat(node, flaw)
        elif flaw_type == PCOPFlawType.OPENCOND:
            return self._get_daughter_nodes_for_opencond(node, flaw)
        else:
            return []

    def execute(self, search_limit=2000):
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
