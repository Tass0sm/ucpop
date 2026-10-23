"""Engine for UCPOP in unified-planning
"""

import unified_planning as up

from ucpop.ucpop import UCPOP
from ucpop.up.pop3_engine import POP3EngineImpl


class UCPOPEngineImpl(POP3EngineImpl):

    @property
    def name(self) -> str:
        return "UCPOP"

    @staticmethod
    def supported_kind():
        supported_kind = POP3EngineImpl.supported_kind()
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UCPOPEngineImpl.supported_kind()

    def _planner(self, problem):
        return UCPOP(problem)


env = up.environment.get_environment()
env.factory.add_engine('ucpop', __name__, 'UCPOPEngineImpl')
