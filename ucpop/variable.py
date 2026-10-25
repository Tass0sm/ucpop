from dataclasses import dataclass, replace
from frozendict import frozendict
from typing import FrozenSet, Union, Tuple, Optional

from unified_planning.model import OperatorKind, Effect, FNode, Type

from ucpop.utils import effect_to_conjunct


@dataclass(eq=True, frozen=True)
class Var:
    name: str          # The variable's name
    type: Type
    num: int


def is_choice(x) -> bool:
    """Whether X is a choice variable, which selects one of several
    alternative ways a plan resolves a threat. Its type is an integer range
    starting at 1, since a constant must be truthy."""
    return isinstance(x, Var) and x.type.is_int_type()


def domain(problem, var: Var) -> list:
    """The values VAR may be bound to: the objects of its type, or the
    integers of a choice variable's range."""
    if is_choice(var):
        return list(range(var.type.lower_bound, var.type.upper_bound + 1))
    return list(problem.objects(var.type))


@dataclass(eq=True, frozen=True)
class UniversalVar(Var):
    """A universally quantified variable of an effect of step NUM. It is never
    added to Bindings: unifying it with a term only selects which instance of
    the effect is meant (section 7.3 of the paper)."""


@dataclass(eq=True, frozen=True)
class BindingNode:
    parent: str
    constant: Optional[str]
    noncodesignation: FrozenSet[str]
    rank: int = 0

    def update(self, parent=None, constant=None, noncodesignation=None, rank=None):
        return BindingNode(parent=parent or self.parent,
                           constant=constant or self.constant,
                           noncodesignation=noncodesignation or self.noncodesignation,
                           rank=rank or self.rank)


Symbol = Union[Var, FNode]
Unifier = list[tuple[Symbol, Symbol, bool]]


@dataclass(eq=True, frozen=True, kw_only=True)
class Bindings:
    nodes: frozendict[Symbol, BindingNode]
    # the set of representative variables that are unbound
    unbound: frozenset[Var]
    size: int
    # each disjunction is a set of alternative conjunctions of constraints,
    # at least one of which must hold
    disjunctions: frozenset[frozenset[frozenset[tuple[Symbol, Symbol, bool]]]] = frozenset()

    @classmethod
    def empty(cls):
        return Bindings(nodes=frozendict(), unbound=frozenset(), size=0)

    def find(self, x: Symbol) -> Symbol:
        return Bindings._find(self.nodes, x)

    def get_grounding(self, x: Symbol) -> Optional[FNode]:
        rx, rep_x = Bindings._get_repr(self.nodes, x)
        return rep_x.constant

    def get_grounding_or_variable(self, x: Symbol) -> Optional[FNode]:
        rx, rep_x = Bindings._get_repr(self.nodes, x)
        return rep_x.constant or rx

    def can_unify(self, x: Symbol, y: Symbol) -> Optional[Unifier]:
        return self._can_unify(self.nodes, x, y)

    def can_divorce(self, x: Symbol, y: Symbol) -> Optional[Unifier]:
        return self._can_divorce(self.nodes, x, y)

    def assume(self, constraints: Unifier) -> Optional["Bindings"]:
        """These bindings with CONSTRAINTS added, or None if they are
        inconsistent with them or leave a disjunction no alternative."""
        bindings = self._assume_constraints(constraints)
        return bindings._propagate() if bindings is not None else None

    def _assume_constraints(self, constraints: Unifier) -> Optional["Bindings"]:
        bindings = self
        for x, y, eq in constraints:
            consistent, unifier = bindings.can_unify(x, y) if eq else bindings.can_divorce(x, y)
            if not consistent:
                return None
            bindings = bindings.union(new_constraints=unifier)
        return bindings

    def _propagate(self) -> Optional["Bindings"]:
        """These bindings without the disjunctions they satisfy and with the
        only alternative of a disjunction that has one left added, or None if
        a disjunction has none. Each alternative is only checked against the
        bindings on its own, so this may miss that they conflict."""
        kept = []
        for disjunction in self.disjunctions:
            if any(self.entails(a) for a in disjunction):
                continue
            alive = [a for a in disjunction if self._assume_constraints(a) is not None]
            if not alive:
                return None
            if len(alive) == 1:
                rest = replace(self, disjunctions=self.disjunctions - {disjunction})
                return rest.assume(alive[0])
            kept.append(frozenset(alive))
        return replace(self, disjunctions=frozenset(kept))

    def with_disjunction(self, alternatives: list[Unifier]) -> Optional["Bindings"]:
        """These bindings requiring one of ALTERNATIVES, each a conjunction
        of constraints, to hold, or None if none can."""
        disjunction = frozenset(frozenset(a) for a in alternatives)
        return replace(self, disjunctions=self.disjunctions | {disjunction}, size=self.size + 1)._propagate()

    def entails(self, constraints: Unifier) -> bool:
        """Whether every one of CONSTRAINTS holds under every grounding of
        these bindings. Sound but, as with can_unify, blind to the finiteness
        of the domains."""
        return all(self.find(x) == self.find(y) if eq else not self.can_unify(x, y)[0]
                   for x, y, eq in constraints)

    def commitments(self) -> int:
        """How much these bindings commit to: one for each variable merged
        into another variable or object, each variable bound to an object,
        and each noncodesignation. Disjunctions are not counted, so linking
        under one costs less than committing to any of its alternatives."""
        # counted twice, since each noncodesignation is in both of its roots
        n = 0
        for x, node in self.nodes.items():
            if self.find(x) != x:
                n += 2 * isinstance(x, Var)
            else:
                n += 2 * (isinstance(x, Var) and node.constant is not None) + len(node.noncodesignation)
        return n // 2

    def union(self, new_variables: list[Var] = [], new_constraints: Unifier = []) -> "Bindings":
        mutable_nodes = dict(self.nodes)

        # Add new variables
        for var in new_variables:
            assert isinstance(var, Var), "new_variables argument must only contain variables"
            mutable_nodes[var] = Bindings._make_node(var)

        # The new variables
        mutable_unbound = set(self.unbound) | set(new_variables)

        for x, y, eq in new_constraints:
            if eq:
                Bindings._add_codesignation(mutable_nodes, mutable_unbound, x, y)
            else:
                Bindings._add_noncodesignation(mutable_nodes, mutable_unbound, x, y)

        return replace(self, nodes=frozendict(mutable_nodes), unbound=frozenset(mutable_unbound), size=self.size + len(new_constraints))

    @staticmethod
    def _find(nodes: dict, x: Symbol) -> Symbol:
        """Find representative of the set containing `name`."""
        if x in nodes:
            while nodes[x].parent != x:
                x = nodes[x].parent
        return x

    @staticmethod
    def _get_repr(nodes: dict, x: Symbol) -> Tuple[Symbol, BindingNode]:
        root = Bindings._find(nodes, x)
        return root, nodes.get(root, Bindings._make_node(x))

    @staticmethod
    def _can_unify(nodes: dict, x: Symbol, y: Symbol) -> Optional[Unifier]:
        rx, rep_x = Bindings._get_repr(nodes, x)
        ry, rep_y = Bindings._get_repr(nodes, y)

        if rx == ry:
            return True, []
        if (rep_x.constant and rep_y.constant and
                rep_x.constant != rep_y.constant):
            return False, None
        if rx in rep_y.noncodesignation or ry in rep_x.noncodesignation:
            return False, None
        return True, [(x, y, True)]

    @staticmethod
    def _can_divorce(nodes: dict, x: Symbol, y: Symbol) -> Optional[Unifier]:
        rx, rep_x = Bindings._get_repr(nodes, x)
        ry, rep_y = Bindings._get_repr(nodes, y)

        if rx == ry:
            return False, None
        if (rep_x.constant and rep_y.constant and
                rep_x.constant != rep_y.constant):
            return True, []
        if rx in rep_y.noncodesignation or ry in rep_x.noncodesignation:
            return True, []
        return True, [(x, y, False)]

    @staticmethod
    def _make_node(x: Symbol):
        return BindingNode(parent=x, constant=(None if isinstance(x, Var) else x),
                           noncodesignation=frozenset(), rank=0)

    @staticmethod
    def _add_codesignation(nodes: dict, unbound: set, x: Symbol, y: Symbol) -> Optional["Bindings"]:
        """Should only be used when x and y can be made to codesignate."""
        rx, rep_x = Bindings._get_repr(nodes, x)
        ry, rep_y = Bindings._get_repr(nodes, y)

        if rx == ry:
            return nodes
        if (rep_x.constant and rep_y.constant and
            rep_x.constant != rep_y.constant):
            # TODO: Maybe return error
            return None
        if rx in rep_y.noncodesignation or ry in rep_x.noncodesignation:
            # TODO: Maybe return error
            return None

        # Choose new root
        if rep_x.rank > rep_y.rank:
            new_root, child = rx, ry
            root_node, child_node = rep_x, rep_y
        else:
            new_root, child = ry, rx
            root_node, child_node = rep_y, rep_x

        # Merge noncodesignation sets
        new_ncd = rep_x.noncodesignation | rep_y.noncodesignation
        new_const = root_node.constant or child_node.constant

        # remove the old child and parent items from the unbound set because
        # they will both be replaced.
        unbound.discard(child)
        unbound.discard(new_root)

        # Update child node
        nodes[child] = child_node.update(parent=new_root,
                                         constant=new_const,
                                         noncodesignation=new_ncd)

        # Update root node
        new_rank = max(rep_x.rank, rep_y.rank) + int(rep_x.rank == rep_y.rank)
        nodes[new_root] = root_node.update(parent=new_root,
                                           constant=new_const,
                                           noncodesignation=new_ncd,
                                           rank=new_rank)

        # if new const is None, then new_root should be a Var
        if new_const is None:
            # double check just in case, though
            assert isinstance(new_root, Var), "Trying to add a constant to the unbound set."
            unbound |= {new_root}


        # Update all former noncodesignation refs to point to new root
        # consider the situation:
        # x y z w
        # and constraints ((neq y w)) meaning y.ncd = {w} and w.ncd = {y}
        # without this step and with unioning (x, y) and (z, w)
        # x-y z-w with x.ncd = {w}, y.ncd = {w}, z.ncd={y}, and w.ncd={y}
        # then union on x and z would check if x is in z.ncd and if z is in x.ncd and find (False or False)
        # then x-(y, z-w) with x.ncd = {w, y} but they are all codesignated so this would be inconsistent.
        # so this is necessary.
        # that would make:
        # - union(x, y) update w.ncd to contain just the representative for y (x)
        # - union(z, w) update y.ncd to contain just the representative for w (z)
        for other in new_ncd:
            other_node = nodes[other]
            nodes[other] = other_node.update(noncodesignation=((other_node.noncodesignation - {rx, ry}) | {new_root}))

    @staticmethod
    def _add_noncodesignation(nodes: dict, unbound: set, x: Symbol, y: Symbol) -> Optional["Bindings"]:
        """Should only be used when x and y can be made to codesignate."""
        rx, rep_x = Bindings._get_repr(nodes, x)
        ry, rep_y = Bindings._get_repr(nodes, y)

        if rx == ry:
            return None
        
        nodes[rx] = rep_x.update(noncodesignation=rep_x.noncodesignation | {ry})
        nodes[ry] = rep_y.update(noncodesignation=rep_y.noncodesignation | {rx})


def most_general_unification(q: FNode, q_step_id: int, r: Union[FNode, Effect], r_step_id: int, bindings: Bindings = {}) -> Optional[Unifier]:
    """Test if the parameterized effect R from step STEP_ID can be unified with
    the condition Q under constraints imposed by BINDINGS. If r is an FNode, its
    interpreted as r := True.
    """

    if isinstance(r, Effect):
        r = effect_to_conjunct(r)

    if q.is_not() != r.is_not():
        return None

    if q.is_not() and r.is_not():
        q = ~q
        r = ~r

    if q.fluent() == r.fluent():
        return unify_args(q.args, q_step_id, r.args, r_step_id, bindings)
    else:
        return None


def to_symbol(x: FNode, step_id: int) -> Symbol:
    if x.node_type == OperatorKind.PARAM_EXP:
        return Var(x.parameter().name, x.parameter().type, step_id)
    elif x.node_type == OperatorKind.VARIABLE_EXP:
        return UniversalVar(x.variable().name, x.variable().type, step_id)
    elif x.node_type == OperatorKind.OBJECT_EXP:
        return x.constant_value()
    return x


def unify_args(q_args: tuple[FNode, ...], q_step_id: int, r_args: tuple[FNode, ...], r_step_id: int, bindings) -> Optional[Unifier]:
    result = []
    # the term each universal variable has been unified with so far
    instance = {}

    for x, y in zip(q_args, r_args):
        x = to_symbol(x, q_step_id)
        y = to_symbol(y, r_step_id)

        if isinstance(x, UniversalVar):
            x, y = y, x
        if isinstance(y, UniversalVar):
            if y in instance:
                # a repeated universal variable must match the same term
                y = instance[y]
            elif y.type.is_compatible(x.type):
                instance[y] = x
                result.append((x, y, True))
                continue
            else:
                return None

        can_unify, unifier = bindings.can_unify(x, y)
        if can_unify:
            result.extend(unifier)
        else:
            return None

    return result


def split_universal(unifier: Unifier) -> Tuple[Unifier, dict[UniversalVar, Symbol]]:
    """Split UNIFIER into its ordinary codesignations and the instance it
    selects, i.e. the term each universal variable was unified with."""
    ordinary, instance = [], {}
    for x, y, eq in unifier:
        if isinstance(y, UniversalVar):
            instance[y] = x
        else:
            ordinary.append((x, y, eq))
    return ordinary, instance
