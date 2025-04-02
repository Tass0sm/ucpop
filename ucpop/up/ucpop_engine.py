"""Engine for UCPOP in unified-planning
"""

import random
from typing import Callable, IO, Optional
from functools import reduce

import unified_planning as up
from unified_planning import engines
from unified_planning.plans import PartialOrderPlan


class UCPOPEngineImpl(up.engines.Engine,
                      up.engines.mixins.OneshotPlannerMixin):

    def __init__(self, **options):
        # Read known user-options and store them for using in the `solve` method
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        # self.max_tries = options.get('max_tries', None)
        # self.restart_probability = options.get('restart_probability', 0.00001)

    @property
    def name(self) -> str:
        return "UCPOP"

    @staticmethod
    def supported_kind():
        # For this demo we limit ourselves to numeric planning.
        # Other kinds of problems can be modeled in the UP library,
        # see unified_planning.model.problem_kind.
        supported_kind = up.model.ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_typing('HIERARCHICAL_TYPING')
        supported_kind.set_numbers('CONTINUOUS_NUMBERS')
        supported_kind.set_numbers('DISCRETE_NUMBERS')
        supported_kind.set_fluents_type('NUMERIC_FLUENTS')
        supported_kind.set_numbers('BOUNDED_TYPES')
        supported_kind.set_fluents_type('OBJECT_FLUENTS')
        supported_kind.set_conditions_kind('NEGATIVE_CONDITIONS')
        supported_kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS')
        supported_kind.set_conditions_kind('EQUALITIES')
        supported_kind.set_conditions_kind('EXISTENTIAL_CONDITIONS')
        supported_kind.set_conditions_kind('UNIVERSAL_CONDITIONS')
        supported_kind.set_effects_kind('CONDITIONAL_EFFECTS')
        supported_kind.set_effects_kind('INCREASE_EFFECTS')
        supported_kind.set_effects_kind('DECREASE_EFFECTS')
        supported_kind.set_effects_kind('FLUENTS_IN_NUMERIC_ASSIGNMENTS')

        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UCPOPEngineImpl.supported_kind()

    def _solve(
            self,
            problem: 'up.model.Problem',
            callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
            timeout: Optional[float] = None,
            output_stream: Optional[IO[str]] = None
    ) -> 'up.engines.PlanGenerationResult':
        env = problem.environment

        # # First we ground the problem
        # with env.factory.Compiler(problem_kind=problem.kind, compilation_kind=up.engines.CompilationKind.GROUNDING) as grounder:
        #     grounding_result = grounder.compile(problem, up.engines.CompilationKind.GROUNDING)
        # grounded_problem = grounding_result.problem

        # # We store the grounded actions in a list
        # actions = list(grounded_problem.instantaneous_actions)

        # aima_problem = self._convert_problem(problem)
        # planner = PartialOrderPlanner(aima_problem)
        # constraints, causal_links = planner.execute(display=False)

        goal = reduce(lambda x, y: x & y, problem.goals)
        plan, info = ucpop.plan(problem, problem.initial_values, goal)

        status = up.engines.PlanGenerationResultStatus.TIMEOUT
        return up.engines.PlanGenerationResult(status, None, self.name)

        # # The candidate plan, initially empty
        # plan = up.plans.SequentialPlan([])

        # # Ask for an instance of a PlanValidator by name
        # # (`sequential_plan_validator` is a python implementation of the
        # # PlanValidator operation mode offered by the UP library)
        # with env.factory.PlanValidator(name='sequential_plan_validator') as pv:
        #     counter = 0
        #     while True:
        #         # With a certain probability, restart from scratch to avoid dead-ends
        #         if random.random() < self.restart_probability:
        #             plan = up.plans.SequentialPlan([])
        #         else:
        #             # Select a random action
        #             a = random.choice(actions)
        #             # Create the relative action instance
        #             ai = up.plans.ActionInstance(a)
        #             # Append the action to the plan
        #             plan.actions.append(ai)

        #             # Check plan validity
        #             res = pv.validate(grounded_problem, plan)
        #             if res:
        #                 # If the plan is valid, lift the action instances and
        #                 # return the resulting plan
        #                 resplan = plan.replace_action_instances(grounding_result.map_back_action_instance)
        #                 # Sanity check
        #                 assert pv.validate(problem, resplan)
        #                 status = up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING
        #                 return up.engines.PlanGenerationResult(status, resplan, self.name)
        #             else:
        #                 # If the plan is invalid, check if the reason is action
        #                 # applicability (as opposed to goal satisfaction)
        #                 einfo = res.log_messages[0].message
        #                 if 'Goals' not in einfo:
        #                     # If the plan is not executable, remove the last action
        #                     plan.actions.pop()
        #             # Limit the number of tries, according to the user specification
        #             counter += 1
        #             if self.max_tries is not None and counter >= self.max_tries:
        #                 status = up.engines.PlanGenerationResultStatus.TIMEOUT
        #                 return up.engines.PlanGenerationResult(status, None, self.name)

    def destroy(self):
        pass


env = up.environment.get_environment()
env.factory.add_engine('ucpop', __name__, 'UCPOPEngineImpl')
