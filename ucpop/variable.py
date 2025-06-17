from dataclasses import dataclass
from frozendict import frozendict
from typing import FrozenSet, Union, Tuple, Optional

from unified_planning.model import OperatorKind, Effect, FNode, Type

from ucpop.utils import effect_to_conjunct


@dataclass(eq=True, frozen=True)
class Var:
    name: str          # The variable's name
    type: Type
    num: int


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


@dataclass(eq=True, frozen=True)
class Bindings:
    nodes: frozendict[Symbol, BindingNode]
    # the set of representative variables that are unbound
    unbound: frozenset[Var]
    size: int

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

        return Bindings(frozendict(mutable_nodes), frozenset(mutable_unbound), self.size + len(new_constraints))

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

    # @staticmethod
    # def from_terms(terms: Set[str], constants: Set[str] = set()) -> "ConstraintState":
    #     nodes = {
    #         t: NodeState(
    #             parent=t,
    #             constant=t if t in constants else None,
    #             noncodesignation=frozenset(),
    #             rank=0
    #         )
    #         for t in terms
    #     }
    #     return ConstraintState(frozendict(nodes))


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


def unify_args(q_args: tuple[FNode, ...], q_step_id: int, r_args: tuple[FNode, ...], r_step_id: int, bindings) -> Optional[Unifier]:
    result = []

    for x, y in zip(q_args, r_args):

        if x.node_type == OperatorKind.PARAM_EXP:
            x = Var(x.parameter().name, x.parameter().type, q_step_id)
        elif x.node_type == OperatorKind.OBJECT_EXP:
            x = x.constant_value()
        if y.node_type == OperatorKind.PARAM_EXP:
            y = Var(y.parameter().name, y.parameter().type, r_step_id)
        elif y.node_type == OperatorKind.OBJECT_EXP:
            y = y.constant_value()

        can_unify, unifier = bindings.can_unify(x, y)
        if can_unify:
            result.extend(unifier)
        else:
            return None

    return result
