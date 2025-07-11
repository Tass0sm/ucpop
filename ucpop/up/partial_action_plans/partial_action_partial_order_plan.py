import networkx as nx
import unified_planning as up
import unified_planning.plans as plans
from unified_planning.environment import Environment
from unified_planning.exceptions import UPUsageError
from unified_planning.plans.plan import ActionInstance
from unified_planning.plans.sequential_plan import SequentialPlan
from typing import Callable, Dict, Iterator, List, Optional


class PartialActionPartialOrderPlan(plans.plan.Plan):
    """Represents a partial order plan. Actions are represent as an adjacency list graph."""

    def __init__(
            self,
            adjacency_dicts: dict[
                "PartialActionInstance", dict["PartialActionInstance", dict[str, any]]
            ],
            relevant_variable_bindings: dict[any, list[any]],
            environment: Optional["Environment"] = None,
            _graph: Optional[nx.DiGraph] = None,
    ):
        """
        Constructs the PartialActionPartialOrderPlan using the labeled adjacency list representation.

        :param adjacency_dicts: The Dictionary representing the adjacency dicts for this PartialOrderPlan.
        :param environment: The environment in which the ActionInstances in the adjacency_dicts are created.
        :param _graph: The graph that is semantically equivalent to the adjacency_dicts.
            NOTE: This parameter is for internal use only and it's maintainance is not guaranteed by any means.
        :return: The created PartialOrderPlan.
        """

        self._relevant_variable_bindings = relevant_variable_bindings

        # if we have a specific environment or we don't have any actions
        if environment is not None or not adjacency_dicts:
            plans.plan.Plan.__init__(
                self, plans.plan.PlanKind.PARTIAL_ORDER_PLAN, environment
            )
        # If we don't have a specific environment, use the environment of the first action
        else:
            assert len(adjacency_dicts) > 0
            for ai in adjacency_dicts.keys():
                plans.plan.Plan.__init__(
                    self, plans.plan.PlanKind.PARTIAL_ORDER_PLAN, ai.action.environment
                )
                break
        if _graph is not None:
            # sanity checks
            assert len(adjacency_dicts) == 0
            assert all(isinstance(n, ActionInstance) for n in _graph.nodes)
            assert all(
                isinstance(f, ActionInstance) and isinstance(t, ActionInstance)
                for f, t in _graph.edges
            )
            self._graph = _graph
        else:
            for (
                ai_k,
                ai_v_dict,
            ) in (
                adjacency_dicts.items()
            ):  # check that given environment and the environment in the actions is the same
                if ai_k.action.environment != self._environment:
                    raise UPUsageError(
                        "The environment given to the plan is not the same of the actions in the plan."
                    )
                for ai in ai_v_dict.keys():
                    if ai.action.environment != self._environment:
                        raise UPUsageError(
                            "The environment given to the plan is not the same of the actions in the plan."
                        )
            self._graph = nx.convert.from_dict_of_dicts(
                adjacency_dicts, create_using=nx.DiGraph
            )

    def __repr__(self) -> str:
        return f"PartialActionPartialOrderPlan({repr(self.get_adjacency_dicts)})"

    def __str__(self) -> str:
        ret = ["PartialActionPartialOrderPlan:", "  actions:"]

        # give an ID, starting from 0, to every ActionInstance in the Plan
        swap_couple = lambda x: (x[1], x[0])
        id: Dict[ActionInstance, int] = dict(
            map(swap_couple, enumerate(nx.topological_sort(self._graph)))
        )
        convert_action_id = lambda action_id: f"    {action_id[1]}) {action_id[0]}"
        ret.extend(map(convert_action_id, id.items()))

        ret.append("  constraints:")
        adj_dicts = self.get_adjacency_dicts

        def convert_action_adj_dict(action_adj_dict):
            action = action_adj_dict[0]
            adj_dict = action_adj_dict[1]
            adj_list = [(str(id[ai]), labels["condition"]) for ai, labels in adj_dict.items()]
            adj_list_str = ", ".join(map(lambda x: x[0] + (" if " + str(x[1])) if x[1] else "", adj_list))
            return f"    {id[action]} < {adj_list_str}"

        ret.extend(
            map(
                convert_action_adj_dict,
                ((act, adj_dict) for act, adj_dict in adj_dicts.items() if adj_dict),
            )
        )

        ret.append("  bindings:")

        def print_bindings(var_bindings):
            var, (possible_bindings, invalid_bindings) = var_bindings
            bindings_str = ", ".join(map(str, possible_bindings))
            if invalid_bindings:
                bindings_str += " and can't be " + ", ".join(map(str, invalid_bindings))
            return f"    {var}: {bindings_str}"

        ret.extend(map(print_bindings, self._relevant_variable_bindings.items()))

        return "\n".join(ret)

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, PartialActionPartialOrderPlan):
            return nx.is_isomorphic(
                self._graph,
                oth._graph,
                node_match=_semantically_equivalent_action_instances,
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash(nx.weisfeiler_lehman_graph_hash(self._graph))

    def __contains__(self, item: object) -> bool:
        if isinstance(item, ActionInstance):
            return any(item.is_semantically_equivalent(a) for a in self._graph.nodes)
        else:
            return False

    @property
    def get_adjacency_dicts(
        self,
    ) -> dict["PartialActionInstance", dict["PartialActionInstance", dict[str, any]]]:
        """Returns the graph of action instances as an adjacency list."""
        return nx.convert.to_dict_of_dicts(self._graph)

    def replace_action_instances(
        self,
        replace_function: Callable[
            ["plans.plan.ActionInstance"], Optional["plans.plan.ActionInstance"]
        ],
    ) -> "plans.plan.Plan":
        """
        Returns a new `PartialActionPartialOrderPlan` where every `ActionInstance` of the current plan is replaced using the given `replace_function`.

        :param replace_function: The function that applied to an `ActionInstance A` returns the `ActionInstance B`; `B`
            replaces `A` in the resulting `Plan`.
        :return: The `PartialActionPartialOrderPlan` where every `ActionInstance` is replaced using the given `replace_function`.
        """
        # first replace all nodes and store the mapping, then use the mapping to
        # recreate the adjacency list representing the new graph
        # ai = action_instance
        original_to_replaced_ai: Dict[
            "plans.plan.ActionInstance", "plans.plan.ActionInstance"
        ] = {}
        for ai in self._graph.nodes:
            replaced_ai = replace_function(ai)
            if replaced_ai is not None:
                original_to_replaced_ai[ai] = replaced_ai

        new_adj_list: Dict[
            "plans.plan.ActionInstance", List["plans.plan.ActionInstance"]
        ] = {}

        # Populate the new adjacency list with the replaced action instances

        for ai in self._graph.nodes:
            replaced_ai = original_to_replaced_ai.get(ai, None)
            if replaced_ai is not None:
                replaced_ai = original_to_replaced_ai[ai]
                replaced_neighbors = []
                for successor in self._graph.neighbors(ai):
                    replaced_successor = original_to_replaced_ai.get(successor, None)
                    if replaced_successor is not None:
                        replaced_neighbors.append(replaced_successor)
                new_adj_list[replaced_ai] = replaced_neighbors

        new_env = self._environment
        for ai in new_adj_list.keys():
            new_env = ai.action.environment
            break
        return up.plans.PartialOrderPlan(new_adj_list, new_env)

    def convert_to(
        self,
        plan_kind: "plans.plan.PlanKind",
        problem: "up.model.AbstractProblem",
    ) -> "plans.plan.Plan":
        """
        This function takes a `PlanKind` and returns the representation of `self`
        in the given `plan_kind`. If the conversion does not make sense, raises
        an exception.

        For the conversion to `SequentialPlan`, returns one  all possible
        `SequentialPlans` that respects the ordering constraints given by
        this `PartialActionPartialOrderPlan`.

        :param plan_kind: The plan_kind of the returned plan.
        :param problem: The `Problem` of which this plan is referring to.
        :return: The plan equivalent to self but represented in the kind of
            `plan_kind`.
        """
        if plan_kind == self._kind:
            return self
        elif plan_kind == plans.plan.PlanKind.SEQUENTIAL_PLAN:
            return SequentialPlan(
                list(nx.topological_sort(self._graph)), self._environment
            )
        else:
            raise UPUsageError(f"{type(self)} can't be converted to {plan_kind}.")

    def all_sequential_plans(self) -> Iterator[SequentialPlan]:
        """Returns all possible `SequentialPlans` that respects the ordering constraints given by this `PartialActionPartialOrderPlan`."""
        for sorted_plan in nx.all_topological_sorts(self._graph):
            yield SequentialPlan(list(sorted_plan), self._environment)

    def get_neighbors(
        self, action_instance: ActionInstance
    ) -> Iterator[ActionInstance]:
        """
        Returns an `Iterator` over all the neighbors of the given `ActionInstance`.

        :param action_instance: The `ActionInstance` of which neighbors must be retrieved.
        :return: The `Iterator` over all the neighbors of the given `action_instance`.
        """
        try:
            retval = self._graph.neighbors(action_instance)
        except nx.NetworkXError:
            raise UPUsageError(
                f"The action instance {str(action_instance)} does not belong to this Partial Action Partial Order Plan. \n Note that 2 Action Instances are equals if and only if they are the exact same object."
            )
        return retval


def _semantically_equivalent_action_instances(
    action_instance_1: ActionInstance, action_instance_2: ActionInstance
) -> bool:
    return action_instance_1.is_semantically_equivalent(action_instance_2)
