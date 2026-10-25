"""Partial-order planning with deferred threat resolution

A plan-space search like POP3's, except that threats are never flaws.
Instead, once a plan has no open goals, the conditions under which every
precondition holds are compiled into a formula over the plan's bindings and
the order of its steps:

    for each precondition q of each step c (and each goal):
        some establisher E (a step with an effect unifying with q under U_E)
        comes before c with U_E holding, and every clobberer t (a step with an
        effect unifying with not q under C_t) has C_t false or comes before E
        or after c.

This is exact for actions without conditional effects: a grounding and a
total order of the steps satisfying the formula is a valid execution, and
every valid execution of the plan satisfies it. So the plan, its bindings
and the formula together are the most general plan for its steps and links,
and whoever executes it may pick any grounding and order satisfying them.

A plan is complete when it has no open goals and the formula is satisfiable
(checked with z3). When it is not, the preconditions in an unsatisfiable
core get another establisher: a new step, since the existing ones already
take part in the formula. Search prefers fewer steps, then fewer binding
commitments. As in PCOP, a goal that several effects of one producer unify
with is linked under the disjunction of their unifiers.

Only actions without conditional or universally quantified effects, and
preconditions and goals that are conjunctions of literals and equalities,
are supported.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import z3
from frozendict import frozendict
from unified_planning.model import FNode, Problem

from ucpop.classes import PlanStep, POP3Plan as Plan
from ucpop.pcop import DisjunctiveLinkNode, DisjunctiveLinking
from ucpop.pop3 import POP3, POP3SearchNode, _effect_sort_key
from ucpop.utils import initial_values_to_conjuncts
from ucpop.variable import Bindings, Var, most_general_unification

logger = logging.getLogger(__name__)


def _supported_goal(q: FNode) -> bool:
    if q.is_and():
        return all(_supported_goal(arg) for arg in q.args)
    if q.is_not():
        q = q.arg(0)
    return q.is_fluent_exp() or q.is_equals()


def check_supported(problem: Problem):
    """Raise ValueError if PROBLEM uses what the deferred planner does not
    support."""
    for action in problem.actions:
        for effect in action.effects:
            if effect.is_conditional() or effect.is_forall():
                raise ValueError(f"{action.name} has a conditional or universally quantified effect")
        if not all(_supported_goal(pc) for pc in action.preconditions):
            raise ValueError(f"{action.name} has a precondition other than a conjunction of literals and equalities")
    if not all(_supported_goal(g) for g in problem.goals):
        raise ValueError("a goal is not a conjunction of literals and equalities")


@dataclass(eq=True, frozen=True)
class Clobberer:
    step: int
    condition: frozenset    # the codesignations under which it clobbers


@dataclass(eq=True, frozen=True)
class Establisher:
    step: int
    condition: frozenset    # the codesignations under which it establishes
    clobberers: tuple[Clobberer, ...]


@dataclass(eq=True, frozen=True)
class Support:
    """Precondition CONDITION of step STEP holds when one of ESTABLISHERS
    comes before STEP under its condition, with none of its clobberers
    coming between them under theirs."""
    condition: FNode
    step: int
    establishers: tuple[Establisher, ...]


def compile_formula(plan: Plan) -> tuple[Support, ...]:
    """The formula, one Support for each precondition of each step that has
    a link. Establishers that cannot come before the step, and clobberers that
    cannot come between, are left out."""
    steps = sorted(plan.steps, key=lambda s: s.id)
    supports = []
    for q, c in sorted({(link.condition, link.step_c) for link in plan.links}, key=lambda p: (p[1].id, p[0].node_id)):
        establishers = []
        for s in steps:
            if s.id in (c.id, -1) or not plan.possibly_before(s, c):
                continue
            for e in sorted(s.effects, key=_effect_sort_key):
                if (unifier := most_general_unification(q, c.id, e, s.id, plan.bindings)) is None:
                    continue
                clobberers = tuple(Clobberer(t.id, frozenset(k))
                                   for t in steps
                                   if t.id not in (s.id, c.id, 0, -1) and plan.possibly_between(t, s, c)
                                   for e_t in sorted(t.effects, key=_effect_sort_key)
                                   if (k := most_general_unification(~q, c.id, e_t, t.id, plan.bindings)) is not None)
                establishers.append(Establisher(s.id, frozenset(unifier), clobberers))
        supports.append(Support(q, c.id, tuple(establishers)))
    return tuple(supports)


def holds(formula: tuple[Support, ...], value, position) -> bool:
    """Whether FORMULA holds when each symbol x is VALUE(x) and each step id
    is at POSITION[id] in a total order."""
    def true(condition):
        return all((value(x) == value(y)) == eq for x, y, eq in condition)

    def protected(e: Establisher, c):
        return not any(true(k.condition) and position[e.step] < position[k.step] < position[c] for k in e.clobberers)

    return all(any(true(e.condition) and position[e.step] < position[sp.step] and protected(e, sp.step)
                   for e in sp.establishers)
               for sp in formula)


class _Encoding:
    """FORMULA as z3 constraints, with a time for each step of PLAN and an
    integer for each unbound variable, indexing the objects of PROBLEM."""

    def __init__(self, plan: Plan, formula: tuple[Support, ...], problem: Problem):
        self.plan, self.problem = plan, problem
        self.index = {o: i for i, o in enumerate(problem.all_objects)}
        self.variables = {}
        self.solver = z3.Solver()
        self.time = {s.id: z3.Int(f"t{s.id}") for s in plan.steps}

        self.solver.add(z3.Distinct(*self.time.values()))
        for u, vs in plan.adj_list.items():
            self.solver.add(*(self.time[u] < self.time[v] for v in vs if v != u))

        bindings = plan.bindings
        for x, node in bindings.nodes.items():
            if bindings.find(x) == x:
                self.solver.add(*(self.value(x) != self.value(y) for y in node.noncodesignation))
        for disjunction in bindings.disjunctions:
            self.solver.add(z3.Or([self.condition(a) for a in disjunction]))

        for i, sp in enumerate(formula):
            self.solver.assert_and_track(self.support(sp), z3.Bool(f"support{i}"))

    def value(self, x):
        bindings = self.plan.bindings
        root = bindings.find(x)
        if (constant := bindings.get_grounding(root)) is not None:
            return z3.IntVal(self.index[constant])
        if root not in self.variables:
            v = self.variables[root] = z3.Int(f"{root.name}{root.num}")
            self.solver.add(z3.Or([v == self.index[o] for o in self.problem.objects(root.type)]))
        return self.variables[root]

    def condition(self, condition):
        return z3.And([(self.value(x) == self.value(y)) if eq else (self.value(x) != self.value(y)) for x, y, eq in condition])

    def support(self, sp: Support):
        t = self.time
        return z3.Or([z3.And(self.condition(e.condition), t[e.step] < t[sp.step],
                             *(z3.Or(z3.Not(self.condition(k.condition)), t[k.step] < t[e.step], t[sp.step] < t[k.step])
                               for k in e.clobberers))
                      for e in sp.establishers])


def _show(x) -> str:
    return f"{x.name}{x.num}" if isinstance(x, Var) else str(x)


def show_plan(plan: Plan, formula: tuple[Support, ...]) -> str:
    """The steps with their parameters, the orderings, and the supports
    that some ordering or binding could break."""
    bindings = plan.bindings
    lines = ["steps:"]
    for s in sorted(plan.steps, key=lambda s: s.id):
        if s.action is not None:
            params = ", ".join(_show(bindings.get_grounding_or_variable(Var(p.name, p.type, s.id))) for p in s.action.parameters)
            lines.append(f"  {s.id}) {s.action.name}({params})")
    edges = sorted((u, v) for u, vs in plan.adj_list.items() for v in vs if u not in (0, -1) and v not in (0, -1))
    lines.append("orderings: " + (", ".join(f"{u} < {v}" for u, v in edges) or "none"))

    def show_condition(condition):
        return " and ".join(f"{_show(bindings.get_grounding_or_variable(x))} {'=' if eq else '!='} {_show(bindings.get_grounding_or_variable(y))}"
                            for x, y, eq in sorted(condition, key=str))

    lines.append("formula:")
    for sp in formula:
        if any(not e.condition and not e.clobberers for e in sp.establishers):
            continue
        lines.append(f"  {sp.condition} at {sp.step} by one of")
        for e in sp.establishers:
            protection = "".join(f"; no {k.step} in between" + (f" where {show_condition(k.condition)}" if k.condition else "")
                                 for k in e.clobberers)
            lines.append(f"    {e.step}" + (f" if {show_condition(e.condition)}" if e.condition else "") + protection)
    return "\n".join(lines)


@dataclass(eq=True, frozen=True)
class DeferredSearchNode(DisjunctiveLinkNode, POP3SearchNode):
    plan: Plan

    @classmethod
    def _make(cls, plan: Plan, agenda):
        return cls(plan, agenda, frozenset())

    def _tie_break_key(self):
        return (len(self.plan.steps), self.plan.bindings.commitments(), len(self.agenda))


class DeferredFlawType(Enum):
    UNPROTECTABLE = 0


class DeferredPOP(DisjunctiveLinking, POP3):

    def __init__(self, problem: Problem):
        check_supported(problem)
        super().__init__(problem)
        self._unprotectable = lru_cache(maxsize=None)(self._unprotectable_supports)

    def _create_initial_node(self) -> DeferredSearchNode:
        start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(self.problem.initial_values))
        end_step = PlanStep(id=-1, preconditions=frozenset(self.problem.goals))
        plan = Plan(steps=frozenset([start_step, end_step]),
                    adj_list=frozendict({0: frozenset({-1})}),
                    bindings=Bindings.empty(),
                    links=frozenset())
        return DeferredSearchNode(plan, frozenset((q, end_step) for q in end_step.preconditions), frozenset())

    def formula(self, plan: Plan) -> tuple[Support, ...]:
        return compile_formula(plan)

    def _unprotectable_supports(self, plan: Plan) -> tuple[Support, ...] | None:
        """The supports in an unsatisfiable core of the plan's formula, or
        None if it is satisfiable. The core is empty when the bindings and
        orderings alone are unsatisfiable, e.g. a link bound a variable
        against a disjunctive binding."""
        formula = compile_formula(plan)
        solver = _Encoding(plan, formula, self.problem).solver
        if solver.check() == z3.sat:
            return None
        core = {str(b) for b in solver.unsat_core()}
        return tuple(sp for i, sp in enumerate(formula) if f"support{i}" in core)

    def _get_flaw(self, node: DeferredSearchNode):
        """Open goals as in POP3, then, once there are none, the supports
        that cannot all hold, which have no resolution if there are none.
        Threats and unbound variables are not flaws."""
        if node.agenda:
            return super()._get_flaw(node)
        if (supports := self._unprotectable(node.plan)) is not None:
            return supports, DeferredFlawType.UNPROTECTABLE
        return None, None

    def _is_complete(self, node: DeferredSearchNode) -> bool:
        return not node.agenda and self._unprotectable(node.plan) is None

    def _get_daughter_nodes_for_flaw(self, node: DeferredSearchNode, flaw_type: Enum, flaw):
        if flaw_type == DeferredFlawType.UNPROTECTABLE:
            return self._get_daughter_nodes_for_unprotectable(node, flaw)
        return super()._get_daughter_nodes_for_flaw(node, flaw_type, flaw)

    def _get_daughter_nodes_for_unprotectable(self, node: DeferredSearchNode, supports):
        """A new step establishing one of the preconditions of SUPPORTS."""
        steps = {s.id: s for s in node.plan.steps}
        children = []
        for sp in supports:
            for action in self.problem.actions:
                for e in sorted(action.effects, key=_effect_sort_key):
                    unifier = most_general_unification(sp.condition, sp.step, e, node.plan.highest_id + 1, node.plan.bindings)
                    if unifier is not None:
                        child, extras = node.with_new_step(action, sp.condition, steps[sp.step], e, unifier)
                        children.append((child, { "note": f"another establisher: {extras['note']}" }))
        return children
