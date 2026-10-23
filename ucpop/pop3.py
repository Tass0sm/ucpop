"""Implementation of POP with conditional effects and disjunction, following
section 6 of https://homes.cs.washington.edu/~weld/papers/pi.pdf

Extends POP2 with:
- conditional effects: the first link from a conditional effect adds its
  antecedent as an open condition, and a threat from a conditional effect can also be
  resolved by confrontation, i.e. adding the negation of its antecedent as an
  open condition of the threatening step;
- disjunctive goals and preconditions: a disjunction on the agenda is
  replaced by one of its disjuncts;
- equality goals: an (in)equality on the agenda, e.g. in an antecedent or a
  disjunct, is not supported by a link but added to the bindings as a
  (non)codesignation constraint, as are an action's top-level ones;

As in POP2, only forced threats are flaws, i.e. the clobbering effect unifies
with the negated link condition without any new codesignations (section 5.1
of the paper), and every variable is eventually grounded, at which point
every remaining threat is forced. Confronting a threat disables the clobbering
effect in that step, so it is not found as a threat again.
"""

import logging
from enum import Enum
from typing import Tuple, FrozenSet, Any
from dataclasses import dataclass, replace

from frozendict import frozendict
from unified_planning.model import FNode, Problem, Action, Effect

from ucpop.search import best_first_search
from ucpop.classes import PlanStep, Link, POP3Plan as Plan
from ucpop.utils import initial_values_to_conjuncts
from ucpop.variable import Bindings, Unifier, most_general_unification, to_symbol
from ucpop.pop2 import POP2SearchNode, POP2

logger = logging.getLogger(__name__)


def _effect_condition(effect) -> FNode | None:
    """The antecedent of a conditional effect, or None if it is unconditional
    (start-step effects are plain FNodes and are always unconditional)."""
    if isinstance(effect, Effect) and effect.is_conditional():
        return effect.condition
    return None


def _effect_sort_key(effect):
    if isinstance(effect, Effect):
        return (effect.fluent.node_id, effect.value.node_id, effect.condition.node_id)
    else:
        return (effect.node_id,)


def _threat_sort_key(threat):
    a_t, link, effect = threat
    return (a_t.id, link.step_p.id, link.condition.node_id, link.step_c.id, _effect_sort_key(effect))


def _conjuncts(q: FNode) -> tuple[FNode, ...] | None:
    """The conjuncts of Q, pushing negations inward, or None if Q is not a
    conjunction."""
    if q.is_and():
        return q.args
    if q.is_not() and (~q).is_or():
        return tuple(~a for a in (~q).args)
    return None


def _equality(q: FNode) -> tuple[FNode, FNode, bool] | None:
    """The terms of Q and whether they must codesignate, or None if Q is not
    an equality or the negation of one."""
    if q.is_equals():
        return (*q.args, True)
    if q.is_not() and (~q).is_equals():
        return (*(~q).args, False)
    return None


def _disjuncts(q: FNode) -> tuple[FNode, ...] | None:
    """The disjuncts of Q, pushing negations inward, or None if Q is not a
    disjunction."""
    if q.is_or():
        return q.args
    if q.is_implies():
        return (~q.arg(0), q.arg(1))
    if q.is_not() and (~q).is_and():
        return tuple(~a for a in (~q).args)
    return None


@dataclass(eq=True, frozen=True)
class POP3SearchNode(POP2SearchNode):
    plan: Plan
    agenda: FrozenSet[Tuple[FNode, PlanStep]]
    threats: FrozenSet[Tuple[PlanStep, Link, Any]]

    # Threats are (threatening step, link, threatening effect). As in POP2,
    # the threats are recomputed by scanning all steps and links after every
    # change to the plan.

    @classmethod
    def _make(cls, plan: Plan, agenda):
        threats = frozenset((step, link, effect)
                            for link in plan.links
                            for step in plan.steps
                            for effect in plan.threatening_effects(step, link))
        return cls(plan, agenda, threats)

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep, effect: Effect, unifier: Unifier):
        new_plan, a_add, new_link, logical_preconds = self.plan.with_new_step(action, q, a_need, unifier)
        new_plan = new_plan.with_used_effect(a_add, effect)

        # add logical preconditions and the antecedent of the linked effect to
        # the agenda
        new_goals = set((pc, a_add) for pc in logical_preconds)
        if (condition := _effect_condition(effect)) is not None:
            new_goals.add((condition, a_add))
        new_agenda = self.agenda - {(q, a_need)} | new_goals

        logger.info(f"Making plan using new action {action.name} with id {a_add.id} and unifier {unifier}")
        return self._make(new_plan, new_agenda), { "note": f"adding {a_add} for {q} in {a_need}" }

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep, effect, unifier: Unifier):
        new_plan, new_link = self.plan.with_reused_step(a_add, q, a_need, unifier)

        # the antecedent is already a goal if the effect supports another link
        new_agenda = self.agenda - {(q, a_need)}
        if (condition := _effect_condition(effect)) is not None and not self.plan.uses(a_add, effect):
            new_agenda = new_agenda | {(condition, a_add)}
        new_plan = new_plan.with_used_effect(a_add, effect)

        logger.info(f"Making plan reusing step: {a_add.action.name if a_add.action else None} with id {a_add.id} and unifier {unifier}")
        return self._make(new_plan, new_agenda), { "note": f"reusing step {a_add} for {q} in {a_need}" }

    def with_new_constraint(self, first_id: int, second_id: int, addressed_threat):
        new_plan = self.plan.with_new_constraint(first_id, second_id)
        return self._make(new_plan, self.agenda), { "note": f"adding {first_id}->{second_id} because {addressed_threat[0]} threatens {addressed_threat[1]}" }

    def with_confrontation(self, addressed_threat):
        a_t, link, effect = addressed_threat
        goal = ~_effect_condition(effect)
        new_plan = self.plan.with_disabled_effect(a_t, effect)
        new_agenda = self.agenda | {(goal, a_t)}
        return self._make(new_plan, new_agenda), { "note": f"confronting with {goal} at {a_t} because it threatens {link}" }

    def with_new_constraints(self, constraints: Unifier, q: FNode, a_need: PlanStep):
        new_plan = replace(self.plan, bindings=self.plan.bindings.union(new_constraints=constraints))
        return self._make(new_plan, self.agenda - {(q, a_need)}), { "note": f"constraining bindings with {q} in {a_need}" }

    def with_new_agenda(self, new_agenda):
        return type(self)(self.plan, new_agenda, self.threats)

    def with_new_binding(self, var, obj):
        new_plan = self.plan.with_new_binding(var, obj)
        return self._make(new_plan, self.agenda), { "note": f"binding {var} to {obj}" }

    def _tie_break_key(self):
        return len(self.plan.steps) + len(self.agenda) + len(self.threats)

    def __le__(self, other):
        return self._tie_break_key() <= other._tie_break_key()

    def __lt__(self, other):
        return self._tie_break_key() < other._tie_break_key()


class POP3FlawType(Enum):
    THREAT = 0
    OPENCOND = 1
    CONJUNCTION_GOAL = 2
    DISJUNCTION_GOAL = 3
    UNBOUND_VAR = 4
    EQUALITY_GOAL = 5


class POP3(POP2):

    def _create_initial_node(self) -> POP3SearchNode:
        # Plan
        start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(self.problem.initial_values))
        end_step = PlanStep(id=-1, preconditions=frozenset(self.problem.goals))

        plan = Plan(steps=frozenset([start_step, end_step]),
                    adj_list=frozendict({0: frozenset({-1})}),
                    bindings=Bindings.empty(),
                    links=frozenset())

        # Agenda
        agenda = frozenset([(q, end_step) for q in end_step.preconditions])

        return POP3SearchNode(plan, agenda, frozenset())

    def _classify_goal(self, q: FNode):
        if _equality(q) is not None:
            return POP3FlawType.EQUALITY_GOAL
        elif _conjuncts(q) is not None:
            return POP3FlawType.CONJUNCTION_GOAL
        elif _disjuncts(q) is not None:
            return POP3FlawType.DISJUNCTION_GOAL
        else:
            return POP3FlawType.OPENCOND

    def _get_flaw(self, node: POP3SearchNode) -> Tuple[Any, Enum]:
        """Threats first, then open conditions, then unbound variables.

        Selection is a deterministic function of plan content, as in PCOP."""
        if node.threats:
            return min(node.threats, key=_threat_sort_key), POP3FlawType.THREAT
        elif node.agenda:
            q, a_c = min(node.agenda, key=lambda t: (t[1].id, t[0].node_id))
            return (q, a_c), self._classify_goal(q)
        elif node.plan.bindings.unbound:
            return min(node.plan.bindings.unbound, key=lambda v: (v.num, v.name)), POP3FlawType.UNBOUND_VAR
        else:
            return None, None

    def _get_daughter_nodes_for_threat(self, node: POP3SearchNode, flaw):
        """Causal link protection: demote, promote or confront."""
        a_t, link, effect = flaw

        new_nodes = []

        if demotion_c := node.plan.can_demote(a_t, link):
            new_nodes.append(node.with_new_constraint(*demotion_c, flaw))

        if promotion_c := node.plan.can_promote(a_t, link):
            new_nodes.append(node.with_new_constraint(*promotion_c, flaw))

        # confrontation: if the clobbering effect is conditional, require its
        # antecedent to be false when A_T executes
        if _effect_condition(effect) is not None:
            new_nodes.append(node.with_confrontation(flaw))

        return new_nodes

    def _get_daughter_nodes_for_opencond(self, node: POP3SearchNode, flaw):
        """Action selection. Unlike POP2, each child remembers which effect
        established the link so that its antecedent can be added as a goal."""
        q, a_need = flaw
        q_step_id = a_need.id

        # step 2(d): the goal already has a link, e.g. when a goal for the
        # same step appeared again. The paper fails here, but the goal may
        # come from a different source (say, a precondition and a conjunct of
        # a quantified one), so like the lisp UCPOP's HANDLE-GOAL, drop it
        if any(link.condition == q and link.step_c.id == a_need.id for link in node.plan.links):
            return [(node.with_new_agenda(node.agenda - {flaw}), { "note": f"{q} in {a_need} is already supported" })]

        def unifying_effects(effects, step_id):
            results = []
            for e in sorted(effects, key=_effect_sort_key):
                unifier = most_general_unification(q, q_step_id, e, step_id, node.plan.bindings)
                if unifier is not None:
                    results.append((e, unifier))
            return results

        # possible plans when a_add is a newly instantiated step
        new_step_plans = [node.with_new_step(action, q, a_need, effect, unifier)
                          for action in self.problem.actions
                          for effect, unifier in unifying_effects(action.effects, node.plan.highest_id + 1)]

        logger.info(f"Getting reusable steps for ({q}, {q_step_id}) under {node.plan.bindings}")

        # possible plans when a_add is a reused step
        reused_step_plans = [node.with_reused_step(step, q, a_need, effect, unifier)
                             for step in sorted(node.plan.steps, key=lambda s: s.id)
                             if node.plan.possibly_before(step, a_need)
                             for effect, unifier in unifying_effects(step.effects, step.id)]

        return new_step_plans + reused_step_plans

    def _get_daughter_nodes_for_conjunction_goal(self, node: POP3SearchNode, flaw):
        q, a_need = flaw
        new_agenda = node.agenda - {(q, a_need)} | set([(arg, a_need) for arg in _conjuncts(q)])
        return [(node.with_new_agenda(new_agenda), { "note": f"expanded {q}" })]

    def _get_daughter_nodes_for_disjunction_goal(self, node: POP3SearchNode, flaw):
        """Replace a disjunction with one of its disjuncts, one child each."""
        q, a_need = flaw
        return [(node.with_new_agenda(node.agenda - {(q, a_need)} | {(d, a_need)}), { "note": f"choosing {d} for {q}" })
                for d in sorted(_disjuncts(q), key=lambda d: d.node_id)]

    def _get_daughter_nodes_for_equality_goal(self, node: POP3SearchNode, flaw):
        """Add an (in)equality to the bindings, or fail if they contradict it."""
        q, a_need = flaw
        x, y, codesignate = _equality(q)
        x, y = to_symbol(x, a_need.id), to_symbol(y, a_need.id)
        bindings = node.plan.bindings
        consistent, constraints = bindings.can_unify(x, y) if codesignate else bindings.can_divorce(x, y)
        if not consistent:
            return []
        return [node.with_new_constraints(constraints, q, a_need)]

    def _get_daughter_nodes_for_flaw(self, node: POP3SearchNode, flaw_type: Enum, flaw):
        if flaw_type == POP3FlawType.THREAT:
            return self._get_daughter_nodes_for_threat(node, flaw)
        elif flaw_type == POP3FlawType.OPENCOND:
            return self._get_daughter_nodes_for_opencond(node, flaw)
        elif flaw_type == POP3FlawType.CONJUNCTION_GOAL:
            return self._get_daughter_nodes_for_conjunction_goal(node, flaw)
        elif flaw_type == POP3FlawType.DISJUNCTION_GOAL:
            return self._get_daughter_nodes_for_disjunction_goal(node, flaw)
        elif flaw_type == POP3FlawType.UNBOUND_VAR:
            return self._get_daughter_nodes_for_unbound_var(node, flaw)
        elif flaw_type == POP3FlawType.EQUALITY_GOAL:
            return self._get_daughter_nodes_for_equality_goal(node, flaw)
        else:
            return []

    def execute(self, search_limit=40000):
        """Run the search in plan-space.
        """

        # step 1
        def pop3_goal_p(node):
            return len(node.agenda) == 0 and len(node.threats) == 0 and len(node.plan.bindings.unbound) == 0

        def pop3_daughters_fn(node):
            # step 2
            flaw, flaw_type = self._get_flaw(node)
            logger.info(f"Addressing {flaw_type} {flaw}")
            # step 3
            daughter_nodes_and_extras = self._get_daughter_nodes_for_flaw(node, flaw_type, flaw)
            logger.info(f"Num Daughters = {len(daughter_nodes_and_extras)}")
            return daughter_nodes_and_extras, { "flaw_type": flaw_type }

        def pop3_rank_fn(node):
            return node._tie_break_key()

        node = self._create_initial_node()
        goal_node, search_tree = best_first_search(node,
                                                   pop3_daughters_fn,
                                                   pop3_goal_p,
                                                   pop3_rank_fn,
                                                   search_limit)

        head = goal_node
        search_path = [goal_node]
        while head in search_tree:
            head, extras = search_tree[head]
            search_path.append((head, extras))
        search_path = list(reversed(search_path))

        return (goal_node.plan, search_path) if goal_node else (None, None)
