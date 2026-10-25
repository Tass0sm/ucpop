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
  in the groundings where C holds. When C entails the codesignations that
  make the white knight's effect the condition, the link and its ordering
  only hold where C does, so nothing is committed; otherwise those
  codesignations are committed and the link always holds.

The threats to a conditional link are those under its condition, and a
conditional link is only a white knight where its condition holds.

When several of the resolutions that only order steps under C apply, i.e.
conditional demotion, promotion and redemption, PCOP does not branch on
them but makes one child with a fresh choice variable b, applying the i-th
under C and b = i. The executor then picks any of them by picking b, and
orderings guarded by different values of b never form a cycle together. A
threat that some values of b resolve remains only under the others, and
separating b from a value drops that alternative.

A goal that several unconditional effects of one producer unify with, say
under U1 and U2, is supported by a single link from that producer with the
disjunctive binding U1 or U2 rather than committing to either. Only
unconditional effects take part, since the antecedent of a conditional one
would have to be a goal only where its unifier holds. The threats to such a
link are found as for any other, since they clobber the consumer's condition
whichever effect supports it.

A plan is complete when it has no open conditions and no threats and its
bindings have at least one grounding.
"""

import logging
from dataclasses import dataclass, replace

from unified_planning.model import Effect
from unified_planning.shortcuts import IntType

from ucpop.classes import PlanStep, PCOPPlan as Plan, link_condition
from ucpop.pop3 import POP3FlawType, _effect_condition, _effect_sort_key, _threat_sort_key
from ucpop.ucpop import UCPOP, UCPOPSearchNode
from ucpop.variable import Bindings, Unifier, UniversalVar, Var, domain, most_general_unification, split_universal

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
        if (bindings := self.plan.bindings.assume([(x, y, False)])) is None:
            return None
        new_plan = replace(self.plan, bindings=bindings)
        return self._make(new_plan, self.agenda), { "note": f"separating {x} != {y} because {addressed_threat[0]} threatens {addressed_threat[1]}" }

    def with_disjunction(self, alternatives: list[Unifier]):
        if (bindings := self.plan.bindings.with_disjunction(alternatives)) is None:
            return None
        return self._make(replace(self.plan, bindings=bindings), self.agenda)

    def with_redemption(self, a_r: PlanStep, effect, unifier: Unifier, addressed_threat, condition):
        a_t, link, _ = addressed_threat
        node, _ = self.with_reused_step(a_r, link.condition, link.step_c, effect, unifier)
        if not node.plan.can_constrain(a_t.id, a_r.id):
            return None
        node, _ = node.with_new_constraint(a_t.id, a_r.id, addressed_threat, condition)
        return node, { "note": f"redeeming {link} with {a_r} after {a_t} if {_show_condition(condition)}" }

    def with_conditional_redemption(self, a_r: PlanStep, effect, unifier: Unifier, addressed_threat, condition):
        """As with_redemption, but the link from A_R and its orderings only
        hold under CONDITION, which entails the ordinary part of UNIFIER, so
        only the instance of a universally quantified effect is committed.
        The antecedent of a conditional effect becomes a goal in every
        grounding, which is stronger than needed."""
        a_t, link, _ = addressed_threat
        _, instance = split_universal(unifier)
        plan, antecedent = self.plan, None
        if not plan.uses(a_r, effect, instance):
            universal = [(x, y, eq) for x, y, eq in unifier if isinstance(y, UniversalVar)]
            plan, constraints, antecedent, pairs = self._instantiate(effect, a_r.id, universal)
            plan = replace(plan, bindings=plan.bindings.union(new_constraints=constraints)).with_used_effect(a_r, effect, pairs)

        if not plan.can_constrain(a_r.id, link.step_c.id, condition):
            return None
        plan = plan.with_conditional_link(a_r, link.condition, link.step_c, condition)
        if not plan.can_constrain(a_t.id, a_r.id, condition):
            return None
        plan = plan.with_new_constraint(a_t.id, a_r.id, condition)
        agenda = self.agenda | {(antecedent, a_r)} if antecedent is not None else self.agenda
        return self._make(plan, agenda), { "note": f"redeeming {link} with {a_r} after {a_t}, both if {_show_condition(condition)}" }

    def _tie_break_key(self):
        """Prefer plans that commit to fewer bindings, then with a shorter
        critical path, then with fewer steps. None of these decreases as a
        plan is refined, so the first complete plan found minimizes them in
        that order. The open goals and threats only break ties."""
        plan = self.plan
        return (plan.bindings.commitments(), plan.critical_path(), len(plan.steps), len(self.agenda) + len(self.threats))


class PCOP(UCPOP):
    plan_class = Plan
    node_class = PCOPSearchNode

    def _get_flaw(self, node: PCOPSearchNode):
        """As in UCPOP, but unbound variables are not flaws, and threats to
        links that differ only in their condition are told apart."""
        if node.threats:
            return min(node.threats, key=lambda t: (_threat_sort_key(t), sorted(map(str, link_condition(t[1]))))), POP3FlawType.THREAT
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
                # every variable is bound, so each disjunction is satisfied or
                # has no alternative left
                return (bindings := bindings.assume([])) is not None and not bindings.disjunctions
            return any(b is not None and search(b, i + 1)
                       for b in (bindings.assume([(unbound[i], o, True)]) for o in domain(self.problem, unbound[i])))

        return search(bindings, 0)

    def _get_daughter_nodes_for_threat(self, node: PCOPSearchNode, flaw):
        a_t, link, effect = flaw
        plan = node.plan
        condition = plan.threat_condition(a_t, link, effect)

        new_nodes = []

        # the resolutions that only order steps where a condition holds, as
        # functions from a node and that condition to a child or None
        def ordering(first, second):
            return lambda n, c: n.with_new_constraint(first, second, flaw, c) if n.plan.can_constrain(first, second, c) else None
        orderings = [ordering(a_t.id, link.step_p.id),          # conditional demotion
                     ordering(link.step_c.id, a_t.id)]          # conditional promotion

        if _effect_condition(effect) is not None:
            new_nodes.append(node.with_confrontation(flaw))

        # separation
        for x, y, _ in sorted(condition, key=str):
            if plan.bindings.can_divorce(x, y)[0] and (child := node.with_separation(x, y, flaw)):
                new_nodes.append(child)

        # redemption by a step that can come after A_T and before the consumer
        for a_r in sorted(plan.steps, key=lambda s: s.id):
            if a_r.id in (a_t.id, link.step_p.id, link.step_c.id) or not plan.can_constrain(a_r.id, link.step_c.id, condition):
                continue
            for e in sorted(a_r.effects, key=_effect_sort_key):
                unifier = most_general_unification(link.condition, link.step_c.id, e, a_r.id, plan.bindings)
                if unifier is None:
                    continue
                ordinary, _ = split_universal(unifier)
                if plan.bindings.assume(condition).entails(ordinary):
                    orderings.append(lambda n, c, a_r=a_r, e=e, unifier=unifier: n.with_conditional_redemption(a_r, e, unifier, flaw, c))
                elif child := node.with_redemption(a_r, e, unifier, flaw, condition):
                    new_nodes.append(child)

        return self._choice_children(node, orderings, condition) + new_nodes

    def _choice_children(self, node: PCOPSearchNode, orderings, condition):
        """The children resolving a threat under CONDITION by ORDERINGS. If
        more than one applies, a single child applies all of them, the i-th
        under CONDITION and a fresh choice variable being i, so it allows
        every one of them rather than committing to one."""
        applicable = [(f, child) for f in orderings if (child := f(node, condition))]
        if len(applicable) < 2:
            return [child for _, child in applicable]

        bindings = node.plan.bindings
        b = Var("choice", IntType(1, len(applicable)), len(bindings.nodes))
        combined = type(node)(replace(node.plan, bindings=bindings.union(new_variables=[b])), node.agenda, node.threats)
        notes = []
        for i, (f, _) in enumerate(applicable, 1):
            if (result := f(combined, condition | {(b, i, True)})) is None:
                return [child for _, child in applicable]
            combined, extras = result
            notes.append(f"{i}: {extras['note']}")
        return [(combined, { "note": f"choosing with {b.name}{b.num} between " + "; ".join(notes) })]

    def _link_children(self, node: PCOPSearchNode, matches, link):
        """As in UCPOP, except that the unconditional effects of the producer
        that need some codesignation to support the goal give a single child,
        linking with the disjunction of their unifiers."""
        def disjunctive(effect, unifier):
            return unifier and _effect_condition(effect) is None and not (isinstance(effect, Effect) and effect.is_forall())

        # an unconditional effect that needs no codesignation already
        # supports the goal without committing to anything
        alternatives = [(e, u) for e, u in matches if disjunctive(e, u)]
        if len(alternatives) < 2 or any(not u and _effect_condition(e) is None for e, u in matches):
            return super()._link_children(node, matches, link)

        children = super()._link_children(node, [(e, u) for e, u in matches if not disjunctive(e, u)], link)
        child, extras = link(alternatives[0][0], [])
        if (child := child.with_disjunction([u for _, u in alternatives])):
            children.append((child, { "note": extras["note"] + " under one of " + " or ".join(_show_condition(u) for _, u in alternatives) }))
        return children
