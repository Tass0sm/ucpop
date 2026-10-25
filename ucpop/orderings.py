"""Compile a deferred plan's formula into gated orderings

An executor like goc-mpc orders steps by edges t(u) <= t(v) that may be
gated by a formula over its own variables: which object each of the plan's
free variables is assigned to, and free binaries it chooses. This module
compiles the formula of a DeferredPOP plan into such edges, exactly: a
grounding and total order of the steps satisfy the formula if and only if
some choice of the binaries satisfies every gated edge whose guard holds.

Supports that always hold are dropped. The rest are compiled in one of two
ways:

- A resource fluent, like (hand_empty ?r), is true initially for every
  object, required by every step that deletes it, and each deleter has a
  restorer that comes after it with the same arguments, like Pick and its
  Place. Its supports hold exactly when, for any two deleters with the same
  arguments, one's restorer comes before the other deleter. So each pair
  gets one binary choosing which goes first.
- Any other support gets one binary per establisher but a fallback (one
  whose condition always holds), selecting it where its condition holds,
  and one binary per clobberer that can come either before the establisher
  or after the consumer, choosing the side.
"""

from dataclasses import dataclass, field
from itertools import combinations, product

from unified_planning.model import Effect, Problem

from ucpop.classes import PlanStep, POP3Plan as Plan
from ucpop.deferred import Support
from ucpop.variable import Bindings, Var, to_symbol


@dataclass(eq=True, frozen=True)
class Same:
    """Variables X and Y are assigned the same object."""
    x: Var
    y: Var


@dataclass(eq=True, frozen=True)
class Assign:
    """Variable X is assigned OBJ."""
    x: Var
    obj: object


@dataclass(eq=True, frozen=True)
class Binary:
    """Free binary INDEX is 1."""
    index: int


@dataclass(eq=True, frozen=True)
class Not:
    arg: object


@dataclass(eq=True, frozen=True)
class And:
    args: tuple


@dataclass(eq=True, frozen=True)
class Or:
    args: tuple


TRUE, FALSE = And(()), Or(())


def conj(*args):
    if any(a == FALSE for a in args):
        return FALSE
    args = tuple(dict.fromkeys(a for a in args if a != TRUE))
    return args[0] if len(args) == 1 else And(args)


def disj(*args):
    if any(a == TRUE for a in args):
        return TRUE
    args = tuple(dict.fromkeys(a for a in args if a != FALSE))
    return args[0] if len(args) == 1 else Or(args)


def neg(a):
    if a == TRUE:
        return FALSE
    if a == FALSE:
        return TRUE
    return a.arg if isinstance(a, Not) else Not(a)


def evaluate(guard, value, bits) -> bool:
    """Whether GUARD holds when each variable x is VALUE(x) and binary i is
    BITS[i]."""
    if isinstance(guard, Same):
        return value(guard.x) == value(guard.y)
    if isinstance(guard, Assign):
        return value(guard.x) == guard.obj
    if isinstance(guard, Binary):
        return bool(bits[guard.index])
    if isinstance(guard, Not):
        return not evaluate(guard.arg, value, bits)
    if isinstance(guard, And):
        return all(evaluate(a, value, bits) for a in guard.args)
    return any(evaluate(a, value, bits) for a in guard.args)


def _variables(guard):
    if isinstance(guard, Same):
        return {guard.x, guard.y}
    if isinstance(guard, Assign):
        return {guard.x}
    if isinstance(guard, Not):
        return _variables(guard.arg)
    if isinstance(guard, (And, Or)):
        return set().union(*(_variables(a) for a in guard.args))
    return set()


@dataclass
class Orderings:
    """The plan's free variables with the objects each may be assigned
    (DOMAINS) and the pairs that must differ (DIFFERENT), the number of free
    binaries, and EDGES: (u, v) step ids -> the guard under which u comes
    before v, TRUE for the plan's own orderings."""
    domains: dict = field(default_factory=dict)
    different: set = field(default_factory=set)
    binaries: int = 0
    edges: dict = field(default_factory=dict)

    def new_binary(self) -> Binary:
        self.binaries += 1
        return Binary(self.binaries - 1)

    def add_edge(self, u, v, guard):
        if guard != FALSE:
            self.edges[(u, v)] = disj(self.edges.get((u, v), FALSE), guard)

    def valid(self, guard) -> bool:
        """Whether GUARD, which has no binaries, holds in every assignment
        of its variables."""
        variables = sorted(_variables(guard), key=lambda v: (v.num, v.name))
        return all(evaluate(guard, dict(zip(variables, values)).get, ())
                   for values in product(*(self.domains[v] for v in variables)))


def _literal(bindings: Bindings, x, y, eq):
    """The guard for the codesignation (x, y, eq), constant if the bindings
    decide it."""
    def term(s):
        root = bindings.find(s)
        constant = bindings.get_grounding(root)
        return root if constant is None else constant

    tx, ty = term(x), term(y)
    if tx == ty:
        return TRUE if eq else FALSE
    if not isinstance(tx, Var) and not isinstance(ty, Var):
        return FALSE if eq else TRUE
    if not bindings.can_unify(tx, ty)[0]:
        return FALSE if eq else TRUE
    if isinstance(tx, Var) and isinstance(ty, Var):
        atom = Same(*sorted((tx, ty), key=lambda v: (v.num, v.name)))
    else:
        atom = Assign(tx, ty) if isinstance(tx, Var) else Assign(ty, tx)
    return atom if eq else neg(atom)


def _condition(bindings: Bindings, condition):
    return conj(*(_literal(bindings, x, y, eq) for x, y, eq in sorted(condition, key=str)))


def _before(plan: Plan, u: int, v: int):
    """TRUE if step U comes before step V in every total order of the plan,
    FALSE if in none, else None."""
    steps = {s.id: s for s in plan.steps}
    if u == v or u == -1 or v == 0:
        return FALSE
    if u == 0 or v == -1 or not plan.possibly_before(steps[v], steps[u]):
        return TRUE
    if not plan.possibly_before(steps[u], steps[v]):
        return FALSE
    return None


def _literal_effect(e):
    """An effect as (fluent expression, whether it makes it true)."""
    if isinstance(e, Effect):
        return e.fluent, e.value.is_true()
    return (e.arg(0), False) if e.is_not() else (e, True)


def _resources(plan: Plan, formula: tuple[Support, ...], problem: Problem):
    """{fluent: [(deleter, restorer, deleted instance)]} for the plan's
    resource fluents."""
    by_fluent = {}
    for sp in formula:
        q = sp.condition
        f = (q.arg(0) if q.is_not() else q).fluent()
        by_fluent.setdefault(f, []).append(sp)

    resources = {}
    for f, supports in by_fluent.items():
        if any(sp.condition.is_not() for sp in supports):
            continue
        if not all(problem.initial_value(f(*objects)).is_true()
                   for objects in product(*(problem.objects(p.type) for p in f.signature))):
            continue

        def changes(s: PlanStep, value):
            return [x for x, v in map(_literal_effect, s.effects) if x.fluent() == f and v == value]

        steps = sorted((s for s in plan.steps if s.id != 0), key=lambda s: s.id)
        deleters = [s for s in steps if changes(s, False)]
        restorers = [s for s in steps if changes(s, True)]
        required = {(sp.step, sp.condition) for sp in supports}
        # each deleter deletes one instance and requires it, and nothing else
        # requires the fluent
        if any(len(changes(d, False)) != 1 or (d.id, changes(d, False)[0]) not in required for d in deleters) or \
           len(required) != len(deleters):
            continue

        def same_instance(d, r):
            x, y = changes(d, False)[0], changes(r, True)[0]
            return _condition(plan.bindings, [(to_symbol(a, d.id), to_symbol(b, r.id), True)
                                              for a, b in zip(x.args, y.args)]) == TRUE

        pairs = []
        for d in deleters:
            matches = [r for r in restorers if len(changes(r, True)) == 1 and same_instance(d, r) and _before(plan, d.id, r.id) == TRUE]
            if len(matches) != 1 or matches[0] in deleters:
                break
            pairs.append((d, matches[0], changes(d, False)[0]))
        if len(pairs) == len(deleters) and len({r for _, r, _ in pairs}) == len(restorers):
            resources[f] = pairs
    return resources


def compile_orderings(plan: Plan, formula: tuple[Support, ...], problem: Problem) -> Orderings:
    """The gated orderings equivalent to FORMULA, the formula of PLAN. Raises
    ValueError if the plan's bindings or formula need what gated orderings
    cannot express: a disjunctive binding over several variables, or a
    support whose establishers' conditions may all fail."""
    bindings = plan.bindings
    orderings = Orderings()
    for v in sorted(bindings.unbound, key=lambda v: (v.num, v.name)):
        orderings.domains[v] = [o for o in problem.objects(v.type) if bindings.assume([(v, o, True)]) is not None]
        orderings.different |= {frozenset((v, w)) for w in bindings.nodes[v].noncodesignation if isinstance(w, Var)}
    if any(len({x for a in d for x, y, _ in a if isinstance(bindings.find(x), Var)} |
               {y for a in d for x, y, _ in a if isinstance(bindings.find(y), Var)}) > 1
           for d in bindings.disjunctions):
        raise ValueError("a disjunctive binding constrains several variables together")

    for u, vs in plan.adj_list.items():
        for v in vs:
            if u not in (0, -1) and v not in (0, -1):
                orderings.add_edge(u, v, TRUE)

    resources = _resources(plan, formula, problem)
    for f, pairs in resources.items():
        for (d_i, r_i, x), (d_j, r_j, y) in combinations(pairs, 2):
            same = _condition(bindings, [(to_symbol(a, d_i.id), to_symbol(b, d_j.id), True) for a, b in zip(x.args, y.args)])
            first, second = _before(plan, r_i.id, d_j.id), _before(plan, r_j.id, d_i.id)
            if same == FALSE or first == TRUE or second == TRUE:
                continue
            if first == FALSE and second == FALSE:
                raise ValueError(f"steps {d_i.id} and {d_j.id} may share {f.name} but can't be ordered either way")
            if first == FALSE:
                orderings.add_edge(r_j.id, d_i.id, same)
            elif second == FALSE:
                orderings.add_edge(r_i.id, d_j.id, same)
            else:
                b = orderings.new_binary()
                orderings.add_edge(r_i.id, d_j.id, conj(same, b))
                orderings.add_edge(r_j.id, d_i.id, conj(same, neg(b)))

    for sp in formula:
        q = sp.condition
        if not q.is_not() and q.fluent() in resources:
            continue
        _compile_support(orderings, plan, sp)
    return orderings


def _compile_support(orderings: Orderings, plan: Plan, sp: Support):
    bindings = plan.bindings
    # establishers from one step with the same clobberers differ only in
    # their condition, e.g. the initial state for each object
    merged = {}
    for e in sp.establishers:
        key = (e.step, e.clobberers)
        merged[key] = disj(merged.get(key, FALSE), _condition(bindings, e.condition))

    # a clobberer that can come neither before the establisher nor after the
    # consumer must not clobber
    candidates = []
    for (step, clobberers), condition in merged.items():
        for k in clobberers:
            if _before(plan, k.step, step) == FALSE and _before(plan, sp.step, k.step) == FALSE:
                condition = conj(condition, neg(_condition(bindings, k.condition)))
        if condition != FALSE and _before(plan, step, sp.step) != FALSE:
            candidates.append((step, clobberers, condition))
    if any(condition == TRUE and not clobberers and _before(plan, step, sp.step) == TRUE
           for step, clobberers, condition in candidates):
        return

    fallbacks = [c for c in candidates if orderings.valid(c[2])]
    if not fallbacks:
        # some establisher's condition must hold: if that is about one
        # variable, it narrows the variable's domain
        required = disj(*(c[2] for c in candidates))
        if len(variables := _variables(required)) == 1:
            (v,) = variables
            orderings.domains[v] = [o for o in orderings.domains[v] if evaluate(required, {v: o}.get, ())]
            fallbacks = [c for c in candidates if orderings.valid(c[2])]
    if not fallbacks:
        raise ValueError(f"{sp.condition} at step {sp.step} has no establisher that always applies")
    fallback = min(fallbacks, key=lambda c: (len(c[1]), c[0]))
    others = [c for c in candidates if c is not fallback]

    # establisher i is selected where its binary is 1 and its condition
    # holds, and no earlier one is; the fallback where none is
    selected = []
    taken = FALSE
    for step, clobberers, condition in others:
        chosen = conj(orderings.new_binary(), condition)
        selected.append(((step, clobberers), conj(chosen, neg(taken))))
        taken = disj(taken, chosen)
    selected.append(((fallback[0], fallback[1]), neg(taken)))

    for (step, clobberers), guard in selected:
        if _before(plan, step, sp.step) is None:
            orderings.add_edge(step, sp.step, guard)
        for k in clobberers:
            clobbers = conj(guard, _condition(bindings, k.condition))
            before, after = _before(plan, k.step, step), _before(plan, sp.step, k.step)
            if before == TRUE or after == TRUE or clobbers == FALSE:
                continue
            if before is None and after is None:
                side = orderings.new_binary()
                orderings.add_edge(k.step, step, conj(clobbers, side))
                orderings.add_edge(sp.step, k.step, conj(clobbers, neg(side)))
            elif before is None:
                orderings.add_edge(k.step, step, clobbers)
            elif after is None:
                orderings.add_edge(sp.step, k.step, clobbers)
