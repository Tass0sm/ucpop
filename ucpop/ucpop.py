"""Implementation of UCPOP following section 7 of
https://homes.cs.washington.edu/~weld/papers/pi.pdf

Extends POP3 (action schemata, conditional effects and disjunction) with
universal quantification. As in the paper, the universe of objects is finite
and static, so quantified goals are replaced with their universal base: a
universally quantified goal becomes the conjunction of its instances, and an
existentially quantified one its body with each variable replaced by a fresh
variable of the step (a Skolem constant). Inside a universal goal, each
instance gets its own Skolem constants. Like any variable, they are bound to
objects of their type before the plan is complete.

Universally quantified effects are not expanded. Instead, each time one is
used to support a link or is confronted, the instance needed is created: the
effect's universal variables get fresh variables of the step, codesignated
with the terms they unified with, and the antecedent and confrontation goals
are posted for that instance only. The antecedent of an instance that already
supports a link is not posted again. A threat from such an effect is forced
when the unifier binds nothing but its universal variables.
"""

import itertools
import logging
from dataclasses import dataclass, replace
from enum import Enum

from frozendict import frozendict
from unified_planning.model import FNode, Action, Effect, Parameter
from unified_planning.shortcuts import And

from ucpop.classes import PlanStep, UCPOPPlan as Plan
from ucpop.pop3 import POP3, POP3SearchNode, _effect_condition
from ucpop.utils import initial_values_to_conjuncts
from ucpop.variable import Bindings, Unifier, Var, UniversalVar, most_general_unification, split_universal

logger = logging.getLogger(__name__)


def _fresh_parameters(variables, bindings: Bindings) -> list[Parameter]:
    """A parameter, with a name not used in BINDINGS, for each of VARIABLES."""
    n = len(bindings.nodes)
    return [Parameter(f"{v.name}#{n + i}", v.type, v.environment) for i, v in enumerate(variables)]


@dataclass(eq=True, frozen=True)
class UCPOPSearchNode(POP3SearchNode):
    plan: Plan

    def _instantiate(self, effect, step_id: int, unifier: Unifier):
        """The instance of EFFECT of step STEP_ID selected by UNIFIER. Each
        universal variable of the effect gets a fresh variable of the step.

        Returns the plan with the fresh variables added, the constraints to
        add (the ordinary part of UNIFIER plus a codesignation for each fresh
        variable of a universal variable the unifier bound), the antecedent
        of the instance (or None), and the instance itself as (universal
        variable, fresh variable) pairs."""
        ordinary, instance = split_universal(unifier)
        if not isinstance(effect, Effect) or not effect.is_forall():
            return self.plan, ordinary, _effect_condition(effect), frozenset()

        em = effect.environment.expression_manager
        fresh_vars, substitution, constraints, pairs = [], {}, list(ordinary), set()
        for v, p in zip(effect.forall, _fresh_parameters(effect.forall, self.plan.bindings)):
            fresh = Var(p.name, p.type, step_id)
            fresh_vars.append(fresh)
            substitution[em.VariableExp(v)] = em.ParameterExp(p)
            if (u := UniversalVar(v.name, v.type, step_id)) in instance:
                constraints.append((fresh, instance[u], True))
                pairs.add((u, fresh))

        plan = replace(self.plan, bindings=self.plan.bindings.union(new_variables=fresh_vars))
        condition = _effect_condition(effect)
        antecedent = condition.substitute(substitution) if condition is not None else None
        return plan, constraints, antecedent, frozenset(pairs)

    def with_new_step(self, action: Action, q: FNode, a_need: PlanStep, effect, unifier: Unifier):
        plan, constraints, antecedent, instance = self._instantiate(effect, self.plan.highest_id + 1, unifier)
        new_plan, a_add, new_link, logical_preconds = plan.with_new_step(action, q, a_need, constraints)
        new_plan = new_plan.with_used_effect(a_add, effect, instance)

        new_goals = set((pc, a_add) for pc in logical_preconds)
        if antecedent is not None:
            new_goals.add((antecedent, a_add))
        new_agenda = self.agenda - {(q, a_need)} | new_goals

        logger.info(f"Making plan using new action {action.name} with id {a_add.id} and unifier {unifier}")
        return self._make(new_plan, new_agenda), { "note": f"adding {a_add} for {q} in {a_need}" }

    def with_reused_step(self, a_add: PlanStep, q: FNode, a_need: PlanStep, effect, unifier: Unifier):
        ordinary, instance = split_universal(unifier)
        if self.plan.uses(a_add, effect, instance):
            # this instance already supports a link, so its antecedent is
            # already a goal
            new_plan, new_link = self.plan.with_reused_step(a_add, q, a_need, ordinary)
            antecedent = None
        else:
            plan, constraints, antecedent, pairs = self._instantiate(effect, a_add.id, unifier)
            new_plan, new_link = plan.with_reused_step(a_add, q, a_need, constraints)
            new_plan = new_plan.with_used_effect(a_add, effect, pairs)

        new_agenda = self.agenda - {(q, a_need)}
        if antecedent is not None:
            new_agenda = new_agenda | {(antecedent, a_add)}

        logger.info(f"Making plan reusing step: {a_add.action.name if a_add.action else None} with id {a_add.id} and unifier {unifier}")
        return self._make(new_plan, new_agenda), { "note": f"reusing step {a_add} for {q} in {a_need}" }

    def with_confrontation(self, addressed_threat):
        a_t, link, effect = addressed_threat
        unifier = most_general_unification(~link.condition, link.step_c.id, effect, a_t.id, self.plan.bindings)
        plan, constraints, antecedent, instance = self._instantiate(effect, a_t.id, unifier)

        new_plan = replace(plan, bindings=plan.bindings.union(new_constraints=constraints))
        new_plan = new_plan.with_disabled_effect(a_t, effect, instance)
        new_agenda = self.agenda | {(~antecedent, a_t)}
        return self._make(new_plan, new_agenda), { "note": f"confronting with {~antecedent} at {a_t} because it threatens {link}" }

    def with_skolem_constants(self, q: FNode, a_need: PlanStep, q_base: FNode, skolems: list[Var]):
        new_plan = replace(self.plan, bindings=self.plan.bindings.union(new_variables=skolems))
        new_agenda = self.agenda - {(q, a_need)} | {(q_base, a_need)}
        return self._make(new_plan, new_agenda), { "note": f"replaced {q} with {q_base}" }


class UCPOPFlawType(Enum):
    UNIVERSAL_GOAL = 0


class UCPOP(POP3):

    def _create_initial_node(self) -> UCPOPSearchNode:
        start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(self.problem.initial_values))
        end_step = PlanStep(id=-1, preconditions=frozenset(self.problem.goals))

        plan = Plan(steps=frozenset([start_step, end_step]),
                    adj_list=frozendict({0: frozenset({-1})}),
                    bindings=Bindings.empty(),
                    links=frozenset())

        agenda = frozenset([(q, end_step) for q in end_step.preconditions])

        return UCPOPSearchNode(plan, agenda, frozenset())

    def _compute_universal_base(self, q: FNode) -> FNode:
        """Replace the outermost (universal) quantifier of Q with the
        conjunction of its instances over the problem's objects. If Q is
        negated, its body is negated in each instance."""
        negated = q.is_not()
        if negated:
            q = ~q

        variables = q.variables()
        instances = [q.arg(0).substitute(dict(zip(variables, objects)))
                     for objects in itertools.product(*[self.problem.objects(v.type) for v in variables])]
        if negated:
            instances = [~i for i in instances]
        return And(instances).simplify()

    def _skolemize(self, q: FNode, bindings: Bindings, step_id: int) -> tuple[FNode, list[Var]]:
        """Replace the variables of the outermost (existential) quantifier of
        Q with fresh variables of step STEP_ID. If Q is negated, its body is
        negated."""
        negated = q.is_not()
        if negated:
            q = ~q

        em = q.environment.expression_manager
        variables = q.variables()
        parameters = _fresh_parameters(variables, bindings)
        body = q.arg(0).substitute({em.VariableExp(v): em.ParameterExp(p) for v, p in zip(variables, parameters)})
        return (~body if negated else body), [Var(p.name, p.type, step_id) for p in parameters]

    def _classify_goal(self, q: FNode):
        if q.is_forall() or q.is_exists() or (q.is_not() and ((~q).is_forall() or (~q).is_exists())):
            return UCPOPFlawType.UNIVERSAL_GOAL
        else:
            return super()._classify_goal(q)

    def _get_daughter_nodes_for_universal_goal(self, node: POP3SearchNode, flaw):
        q, a_need = flaw
        # not forall = exists not, and not exists = forall not
        if q.is_exists() or (q.is_not() and (~q).is_forall()):
            q_base, skolems = self._skolemize(q, node.plan.bindings, a_need.id)
            return [node.with_skolem_constants(q, a_need, q_base, skolems)]

        q_base = self._compute_universal_base(q)
        if q_base.is_false():
            return []
        new_agenda = node.agenda - {(q, a_need)}
        if not q_base.is_true():
            new_agenda = new_agenda | {(q_base, a_need)}
        return [(node.with_new_agenda(new_agenda), { "note": f"replaced {q} with {q_base}" })]

    def _get_daughter_nodes_for_flaw(self, node: POP3SearchNode, flaw_type: Enum, flaw):
        if flaw_type == UCPOPFlawType.UNIVERSAL_GOAL:
            return self._get_daughter_nodes_for_universal_goal(node, flaw)
        else:
            return super()._get_daughter_nodes_for_flaw(node, flaw_type, flaw)
