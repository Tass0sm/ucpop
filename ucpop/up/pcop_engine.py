"""Engine for PCOP in unified-planning
"""

from functools import reduce
from typing import Callable, IO, Optional

import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.model import Variable

from ucpop.up.partial_action_plans import PartialActionInstance, PartialActionPartialOrderPlan
from ucpop.up.ucpop_engine import UCPOPEngineImpl
from ucpop.variable import Var
from ucpop.pcop import PCOP


class PCOPEngineImpl(UCPOPEngineImpl):

    @property
    def name(self) -> str:
        return "PCOP"

    def _planner(self, problem):
        return PCOP(problem)

    @staticmethod
    def _action_adjacency_dicts_from_plan(plan, problem):
        """The plan's steps as partial action instances whose unbound
        parameters are variables, the orderings between them labeled with the
        condition under which they hold (None if always), and for each
        variable the objects it may be bound to and the variables it must
        differ from."""
        id_to_instance_map = {}
        graph = {}
        relevant_variables = {}

        def get_grounding_or_variable(v):
            result = plan.bindings.get_grounding_or_variable(v)
            if isinstance(result, Var):
                if result not in relevant_variables:
                    relevant_variables[result] = Variable(result.name + str(result.num), result.type)
                return relevant_variables[result]
            return result

        for step in sorted(plan.steps, key=lambda s: s.id):
            if step.id in [0, -1]:
                continue
            params = tuple(get_grounding_or_variable(Var(p.name, p.type, step.id)) for p in step.action.parameters)
            action_instance = PartialActionInstance(step.action, params)
            id_to_instance_map[step.id] = action_instance
            graph[action_instance] = {}

        def convert_condition(condition):
            literals = []
            for x, y, eq in sorted(condition, key=str):
                x, y = get_grounding_or_variable(x), get_grounding_or_variable(y)
                literals.append(x.Equals(y) if eq else ~x.Equals(y))
            return reduce(lambda a, b: a & b, literals)

        # an edge holds if it is unconditional or any of its conditions holds
        conditions = {}
        for u, vs in plan.adj_list.items():
            for v in vs:
                conditions[(u, v)] = None
        for u, v, condition in plan.conditional_edges:
            if conditions.get((u, v), False) is not None:
                condition = convert_condition(condition)
                conditions[(u, v)] = condition if (u, v) not in conditions else conditions[(u, v)] | condition

        for (u, v), condition in conditions.items():
            if u in [0, -1] or v in [0, -1] or u == v:
                continue
            graph[id_to_instance_map[u]][id_to_instance_map[v]] = {"condition": condition}

        relevant_variable_bindings = {}
        for var, up_var in list(relevant_variables.items()):
            possible = [o for o in problem.objects(var.type) if plan.bindings.can_unify(var, o)[0]]
            different = [get_grounding_or_variable(other) for other in plan.bindings.nodes[var].noncodesignation
                         if isinstance(other, Var)]
            relevant_variable_bindings[up_var] = (possible, different)

        return graph, relevant_variable_bindings

    def _solve(
            self,
            problem: 'up.model.Problem',
            callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
            timeout: Optional[float] = None,
            output_stream: Optional[IO[str]] = None
    ) -> 'up.engines.PlanGenerationResult':

        plan, _ = self._planner(problem).execute()

        if plan:
            status = PlanGenerationResultStatus.SOLVED_SATISFICING
            action_adjacency_dicts, relevant_variable_bindings = PCOPEngineImpl._action_adjacency_dicts_from_plan(plan, problem)
            return up.engines.PlanGenerationResult(
                status, PartialActionPartialOrderPlan(action_adjacency_dicts, relevant_variable_bindings),
                self.name, metrics={}
            )
        else:
            status = PlanGenerationResultStatus.UNSOLVABLE_PROVEN
            return up.engines.PlanGenerationResult(status, None, self.name)


env = up.environment.get_environment()
env.factory.add_engine('pcop', __name__, 'PCOPEngineImpl')
