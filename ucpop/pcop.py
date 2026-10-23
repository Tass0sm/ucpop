"""Implementation of Partial Conditional Ordering Planning (PCOP)

Extends UCPOP so that a plan is complete without binding all of its
variables. A plan then stands for all of its groundings, i.e. assignments of
objects to its unbound variables that respect its noncodesignations, and it
must be correct in each of them.

UCPOP only treats forced threats as flaws and relies on grounding every
variable to turn the possible threats into forced ones. PCOP treats every
possible threat as a flaw: a step whose effect clobbers a link under some
codesignations C. It can be resolved by
- conditional demotion or promotion: ordering the step before the producer
  or after the consumer in the groundings where C holds;
- separation: a noncodesignation ruling out one of C;
- confrontation, as in UCPOP: requiring the antecedent of a conditional
  effect to be false (in every grounding, which is stronger than needed);
- redemption: reusing a step as a white knight, linking it to the consumer
  for the threatened condition and ordering the threatening step before it
  in the groundings where C holds.

A plan is complete when it has no open conditions and no threats and its
bindings have at least one grounding.
"""

import logging
from dataclasses import dataclass, replace

from ucpop.classes import PlanStep, PCOPPlan as Plan
from ucpop.pop3 import POP3FlawType, _effect_condition, _effect_sort_key
from ucpop.ucpop import UCPOP, UCPOPSearchNode
from ucpop.variable import Bindings, Unifier, UniversalVar, most_general_unification

logger = logging.getLogger(__name__)


def _show_condition(condition):
    return " and ".join(f"{x} {'=' if eq else '!='} {y}" for x, y, eq in condition) or "always"


@dataclass(eq=True, frozen=True)
class PCOPSearchNode(UCPOPSearchNode):
    plan: Plan

    def with_new_constraint(self, first_id: int, second_id: int, addressed_threat, condition=frozenset()):
        new_plan = self.plan.with_new_constraint(first_id, second_id, condition)
        return self._make(new_plan, self.agenda), { "note": f"adding {first_id}->{second_id} if {_show_condition(condition)} because {addressed_threat[0]} threatens {addressed_threat[1]}" }

    def with_confrontation(self, addressed_threat):
        """As in UCPOP, but only the instance of a universally quantified
        effect is fixed: the threat's other codesignations are not committed."""
        a_t, link, effect = addressed_threat
        unifier = most_general_unification(~link.condition, link.step_c.id, effect, a_t.id, self.plan.bindings)
        instance = [(x, y, eq) for x, y, eq in unifier if isinstance(y, UniversalVar)]
        plan, constraints, antecedent, pairs = self._instantiate(effect, a_t.id, instance)

        new_plan = replace(plan, bindings=plan.bindings.union(new_constraints=constraints))
        new_plan = new_plan.with_disabled_effect(a_t, effect, pairs)
        new_agenda = self.agenda | {(~antecedent, a_t)}
        return self._make(new_plan, new_agenda), { "note": f"confronting with {~antecedent} at {a_t} because it threatens {link}" }

    def with_separation(self, x, y, addressed_threat):
        new_plan = replace(self.plan, bindings=self.plan.bindings.union(new_constraints=[(x, y, False)]))
        return self._make(new_plan, self.agenda), { "note": f"separating {x} != {y} because {addressed_threat[0]} threatens {addressed_threat[1]}" }

    def with_redemption(self, a_r: PlanStep, effect, unifier: Unifier, addressed_threat, condition):
        a_t, link, _ = addressed_threat
        node, _ = self.with_reused_step(a_r, link.condition, link.step_c, effect, unifier)
        if not node.plan.can_constrain(a_t.id, a_r.id):
            return None
        node, _ = node.with_new_constraint(a_t.id, a_r.id, addressed_threat, condition)
        return node, { "note": f"redeeming {link} with {a_r} after {a_t} if {_show_condition(condition)}" }


class PCOP(UCPOP):
    plan_class = Plan
    node_class = PCOPSearchNode

    def _get_flaw(self, node: PCOPSearchNode):
        """As in UCPOP, but unbound variables are not flaws."""
        flaw, flaw_type = super()._get_flaw(node)
        if flaw_type == POP3FlawType.UNBOUND_VAR:
            return None, None
        return flaw, flaw_type

    def _is_complete(self, node: PCOPSearchNode) -> bool:
        return len(node.agenda) == 0 and len(node.threats) == 0 and self._has_grounding(node.plan.bindings)

    def _has_grounding(self, bindings: Bindings) -> bool:
        """Whether some assignment of objects to the unbound variables
        respects the bindings, by backtracking over the objects of each
        variable's type."""
        unbound = sorted(bindings.unbound, key=lambda v: (v.num, v.name))

        def search(bindings, i):
            if i == len(unbound):
                return True
            return any(b is not None and search(b, i + 1)
                       for b in (bindings.assume([(unbound[i], o, True)]) for o in self.problem.objects(unbound[i].type)))

        return search(bindings, 0)

    def _get_daughter_nodes_for_threat(self, node: PCOPSearchNode, flaw):
        a_t, link, effect = flaw
        plan = node.plan
        condition = plan.threat_condition(a_t, link, effect)

        new_nodes = []

        # conditional demotion and promotion
        if plan.can_constrain(a_t.id, link.step_p.id):
            new_nodes.append(node.with_new_constraint(a_t.id, link.step_p.id, flaw, condition))
        if plan.can_constrain(link.step_c.id, a_t.id):
            new_nodes.append(node.with_new_constraint(link.step_c.id, a_t.id, flaw, condition))

        if _effect_condition(effect) is not None:
            new_nodes.append(node.with_confrontation(flaw))

        # separation
        for x, y, _ in sorted(condition, key=str):
            if plan.bindings.can_divorce(x, y)[0]:
                new_nodes.append(node.with_separation(x, y, flaw))

        # redemption by a step that can come after A_T and before the consumer
        for a_r in sorted(plan.steps, key=lambda s: s.id):
            if a_r.id in (a_t.id, link.step_p.id, link.step_c.id) or not plan.possibly_before(a_r, link.step_c):
                continue
            for e in sorted(a_r.effects, key=_effect_sort_key):
                unifier = most_general_unification(link.condition, link.step_c.id, e, a_r.id, plan.bindings)
                if unifier is not None and (child := node.with_redemption(a_r, e, unifier, flaw, condition)):
                    new_nodes.append(child)

        return new_nodes
