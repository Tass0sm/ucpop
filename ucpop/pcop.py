"""Implementation of Partial Conditional Ordering Planning
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""

import random
import logging
import itertools
from enum import Enum
from typing import List, Dict, Tuple, Collection, FrozenSet, Any
from functools import reduce
from dataclasses import dataclass

from frozendict import frozendict
from unified_planning.model import FNode, Problem, Action, Effect, Object
from unified_planning.shortcuts import Exists, Forall

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

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep, all_unifiers: list[Unifier]):
        new_plan, a_add, new_link, logical_preconds = self.plan.with_new_step(action, q, a_need, all_unifiers)

        # add logical preconditions to agenda
        new_agenda = self.agenda - {(q, a_need)} | set([(pc, a_add) for pc in logical_preconds])

        # test all links in new plan with all steps in new plan to identify any new threats due to the new binding
        # O(N^3)
        # TODO: See if this can still be computed incrementally in this case
        added_threats = []
        for link in new_plan.links:
            for step in new_plan.steps.values():
                if (threat_conditions := new_plan.threatens(step, link)) is not None:
                    added_threats.append((step, link, threat_conditions))

        new_threats = self.threats.union(added_threats)

        logger.info(f"Making plan using new action {action.name} with id {a_add.id} and unifiers {all_unifiers}")
        return PCOPSearchNode(new_plan, new_agenda, new_threats), { "note": f"adding {a_add} for {q} in {a_need}" }

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep, all_unifiers: list[Unifier]):
        new_plan, new_link = self.plan.with_reused_step(a_add, q, a_need, all_unifiers)
        new_agenda = self.agenda - {(q, a_need)}

        # test all links in new plan with all steps in new plan to identify any new threats due to the new binding
        # O(N^3)
        # TODO: See if this can still be computed incrementally in this case
        added_threats = []
        for link in new_plan.links:
            for step in new_plan.steps.values():
                if (threat_conditions := new_plan.threatens(step, link)) is not None:
                    added_threats.append((step, link, threat_conditions))

        new_threats = self.threats.union(added_threats)

        logger.info(f"Making plan reusing step: {a_add.action.name if a_add.action else None} with id {a_add.id} under unifiers {all_unifiers}")
        return PCOPSearchNode(new_plan, new_agenda, new_threats), { "note": f"reusing step {a_add} for {q} in {a_need}" }

    def with_new_constraint(
            self,
            first_id: int,
            second_id: int,
            edge_conditions: frozenset[Unifier],
            addressed_threat
    ):
        new_plan = self.plan.with_new_constraint(first_id, second_id, edge_conditions)
        new_threats = self.threats - {addressed_threat}
        return PCOPSearchNode(new_plan, self.agenda, new_threats), { "note": f"adding {first_id}->{second_id} under {edge_conditions} because {addressed_threat[0]} threatens {addressed_threat[1]}" }

    def with_new_bindings(
            self,
            unifier: Unifier,
            addressed_threat: any
    ):
        new_plan = self.plan.with_new_bindings(unifier)

        # test all links in new plan with all steps in new plan to identify any threats
        # O(N^3)
        # TODO: See if this can still be computed incrementally in this case
        new_threats = []
        for link in new_plan.links:
            for step in new_plan.steps.values():
                if (threat_conditions := new_plan.threatens(step, link)) is not None:
                    new_threats.append((step, link, threat_conditions))
        new_threats = frozenset(new_threats)

        return PCOPSearchNode(new_plan, self.agenda, new_threats), { "note": f"adding {unifier} because {addressed_threat[0]} threatens {addressed_threat[1]}" }

    def with_new_agenda(self, new_agenda):
        return PCOPSearchNode(self.plan, new_agenda, self.threats)

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
    UNIVERSAL_GOAL = 2
    CONJUNCTION_GOAL = 3


class PCOP:

    def __init__(self, problem: Problem):
        self.problem = problem

    def _create_initial_node(self) -> PCOPSearchNode:
        # Plan
        start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(self.problem.initial_values))
        end_step = PlanStep(id=-1, preconditions=frozenset(self.problem.goals))

        plan = Plan(steps=frozendict({0: start_step, -1: end_step}),
                    adj_list=frozendict({0: frozenset({(-1, ())})}),
                    bindings=Bindings.empty(),
                    links=frozenset())

        # Agenda
        agenda = frozenset([(q, end_step) for q in end_step.preconditions])

        # Threats
        threats = frozenset([])

        return PCOPSearchNode(plan, agenda, threats)

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

    def _get_flaw(self, node: PCOPSearchNode) -> Tuple[Any, PCOPFlawType]:
        """This corresponds to the goal selection step in the POP
        non-deterministic pseudocode"""
        if node.threats:
            return next(iter(node.threats)), PCOPFlawType.THREAT
        elif node.agenda:
            q, a_c = next(iter(node.agenda))

            if (~q).is_exists() or q.is_exists() or (~q).is_forall() or q.is_forall():
                return (q, a_c), PCOPFlawType.UNIVERSAL_GOAL
            elif q.is_and():
                return (q, a_c), PCOPFlawType.CONJUNCTION_GOAL
            else:
                return (q, a_c), PCOPFlawType.OPENCOND
        else:
            return None, None

    def _get_daughter_nodes_for_threat(self, node: PCOPSearchNode, flaw):
        """This corresponds to the causal link protection step in the POP
        non-deterministic pseudocode"""
        a_t, link, threat_conditions = flaw

        new_nodes = []

        if demotion_c := node.plan.can_demote(a_t, link):
            logger.info("Making plan with demotion")
            new_nodes.append(node.with_new_constraint(*demotion_c, threat_conditions, flaw))

        if promotion_c := node.plan.can_promote(a_t, link):
            logger.info("Making plan with promotion")
            new_nodes.append(node.with_new_constraint(*promotion_c, threat_conditions, flaw))

        # threat conditions is a disjunction of conjunctions. if any clause is
        # true, A_T will threaten LINK. So one only needs to ensure that one
        # requirement from each clause is unsatisfied. If there are N clauses
        # each with M items and you choose to only make disunification per
        # clause, there are M^N possible descendant nodes for an accomodation.

        for requirements in itertools.product(*threat_conditions):
            if unifier := node.plan.can_accommodate(a_t, link, requirements):
                new_nodes.append(node.with_new_bindings(unifier, flaw))

        return new_nodes

    def _get_daughter_nodes_for_opencond(self, node: PCOPSearchNode, flaw):
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

    def _get_daughter_nodes_for_universal_goal(self, node: PCOPSearchNode, flaw):
        """This function is a workaround for the complicated step 2 in UCPOP,
        which resolves a universal goal from the agenda."""
        q, a_need = flaw
        q_base = self._compute_universal_base(q)
        new_agenda = node.agenda - {(q, a_need)} | {(q_base, a_need)}
        return [(node.with_new_agenda(new_agenda), { "note": f"replaced {q} with {q_base}" })]

    def _get_daughter_nodes_for_conjunction_goal(self, node: PCOPSearchNode, flaw):
        """This function is a workaround for the complicated step 2 in UCPOP,
        which resolves a conjunction goal from the agenda."""
        q, a_need = flaw
        new_agenda = node.agenda - {(q, a_need)} | set([(arg, a_need) for arg in q.args])
        return [(node.with_new_agenda(new_agenda), { "note": f"expanded {q}" })]

    def _get_daughter_nodes_for_flaw(self, node: PCOPSearchNode, flaw_type: PCOPFlawType, flaw):
        if flaw_type == PCOPFlawType.THREAT:
            return self._get_daughter_nodes_for_threat(node, flaw)
        elif flaw_type == PCOPFlawType.OPENCOND:
            return self._get_daughter_nodes_for_opencond(node, flaw)
        elif flaw_type == PCOPFlawType.UNIVERSAL_GOAL:
            return self._get_daughter_nodes_for_universal_goal(node, flaw)
        elif flaw_type == PCOPFlawType.CONJUNCTION_GOAL:
            return self._get_daughter_nodes_for_conjunction_goal(node, flaw)
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
            daughter_nodes_and_extras = self._get_daughter_nodes_for_flaw(node, flaw_type, flaw)
            # logger.info(f"Threats: {current.threats}")
            logger.info(f"Num Daughters = {len(daughter_nodes_and_extras)}")

            return daughter_nodes_and_extras, { "flaw_type": flaw_type }

        def pop_rank_fn(node):
            return len(node.plan.steps) + len(node.agenda) + len(node.threats) + node.plan.bindings.size

        node = self._create_initial_node()
        goal_node, search_tree = best_first_search(node,
                                                   pcop_daughters_fn,
                                                   pcop_goal_p,
                                                   pcop_rank_fn,
                                                   search_limit)

        head = goal_node
        search_path = [goal_node]
        while head in search_tree:
            head, extras = search_tree[head]
            search_path.append((head, extras))
        search_path = list(reversed(search_path))

        return (goal_node.plan, search_path) if goal_node else (None, None)
