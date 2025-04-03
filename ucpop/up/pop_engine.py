"""Engine for POP in unified-planning
"""

import random
from collections import defaultdict
from typing import Callable, IO, Optional
from functools import reduce

import unified_planning as up
from unified_planning import engines
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import ActionInstance

from ucpop.pop import pop


class POPEngineImpl(up.engines.Engine,
                    up.engines.mixins.OneshotPlannerMixin):

    def __init__(self, **options):
        # Read known user-options and store them for using in the `solve` method
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        # self.max_tries = options.get('max_tries', None)
        # self.restart_probability = options.get('restart_probability', 0.00001)

    @property
    def name(self) -> str:
        return "POP"

    @staticmethod
    def supported_kind():
        # For this demo we limit ourselves to numeric planning.
        # Other kinds of problems can be modeled in the UP library,
        # see unified_planning.model.problem_kind.
        supported_kind = up.model.ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_fluents_type('OBJECT_FLUENTS')

        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= POPEngineImpl.supported_kind()

    def _action_adjacency_list_from_plan(self, plan):
        id_to_instance_map = {}

        for step in plan.steps:
            if step.id in [0, -1]:
                continue
            id_to_instance_map[step.id] = ActionInstance(step.action)

        graph = defaultdict(list)
        # Build the directed graph
        for a, b in plan.ordering:
            if a in [0, -1] or b in [0, -1] or a == b:
                continue
            a_inst = id_to_instance_map[a]
            b_inst = id_to_instance_map[b]
            print(f"{a_inst} -> {b_inst}")
            graph[a_inst].append(b_inst)

        return graph

    def _solve(
            self,
            problem: 'up.model.Problem',
            callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
            timeout: Optional[float] = None,
            output_stream: Optional[IO[str]] = None
    ) -> 'up.engines.PlanGenerationResult':
        env = problem.environment

        # First we ground the problem
        with env.factory.Compiler(problem_kind=problem.kind, compilation_kind=up.engines.CompilationKind.GROUNDING) as grounder:
            grounding_result = grounder.compile(problem, up.engines.CompilationKind.GROUNDING)
        grounded_problem = grounding_result.problem

        plan, info = pop(grounded_problem)

        if plan:
            status = PlanGenerationResultStatus.SOLVED_SATISFICING
            action_adjacency_list = self._action_adjacency_list_from_plan(plan)
            return up.engines.PlanGenerationResult(
                status, up.plans.PartialOrderPlan(action_adjacency_list),
                self.name, metrics={}
            )
            
        else:
            status = PlanGenerationResultStatus.UNSOLVABLE_PROVEN
            return up.engines.PlanGenerationResult(status, None, self.name)

    def destroy(self):
        pass


env = up.environment.get_environment()
env.factory.add_engine('pop', __name__, 'POPEngineImpl')
