"""Two robots, two items: carry each item from its start point to its goal
point. Reproduces the planning problem of po-goc-mpc's
`open_pick_place_two_robot` experiment (`po_goc_mpc/experiments/pyrobosim/
open_room.py`), without its goc-mpc constraint generators, which don't
affect the search.

PCOP used to return a single chain over one robot variable
(Pick w1 -> Place w1 -> Pick w0 -> Place w0, robot ?r in {robot0, robot1}),
so one robot did all the work. Now that its search prefers fewer binding
commitments and then shorter critical paths, and redeems threats with
links that only hold under the threat's codesignations, it returns one
free robot variable per item, with the two Pick->Place chains ordered only
if both robots are the same, so different robots work in parallel.

Run:
    uv run python scripts/test_open_pick_place.py
"""

import networkx as nx
import unified_planning as up
from unified_planning.shortcuts import (
    BoolType, Fluent, InstantaneousAction, Object, OneshotPlanner, Problem, UserType,
)

import ucpop  # noqa: F401  -- registers the 'pcop' engine

ROBOTS = ["robot0", "robot1"]
ITEMS = ["water0", "water1"]

Point = UserType("Point")
Robot = UserType("Robot")
Item = UserType("Item")

item_at = Fluent("item_at", BoolType(), i=Item, p=Point)
holding = Fluent("holding", BoolType(), r=Robot, i=Item)
hand_empty = Fluent("hand_empty", BoolType(), r=Robot)

pick = InstantaneousAction("Pick", r=Robot, i=Item, p=Point)
r, i, p = pick.parameters
pick.add_precondition(item_at(i, p))
pick.add_precondition(hand_empty(r))
pick.add_effect(item_at(i, p), False)
pick.add_effect(holding(r, i), True)
pick.add_effect(hand_empty(r), False)

place = InstantaneousAction("Place", r=Robot, i=Item, p=Point)
r, i, p = place.parameters
place.add_precondition(holding(r, i))
place.add_effect(holding(r, i), False)
place.add_effect(hand_empty(r), True)
place.add_effect(item_at(i, p), True)

problem = Problem("OpenPickPlace")
problem.add_fluent(item_at, default_initial_value=False)
problem.add_fluent(holding, default_initial_value=False)
problem.add_fluent(hand_empty, default_initial_value=True)
problem.add_action(pick)
problem.add_action(place)

robots = [Object(name, Robot) for name in ROBOTS]
items = {name: Object(name, Item) for name in ITEMS}
points = {f"{name}_{end}": Object(f"{name}_{end}", Point)
          for name in ITEMS for end in ("start", "goal")}
problem.add_objects([*robots, *items.values(), *points.values()])
for name, item in items.items():
    problem.set_initial_value(item_at(item, points[f"{name}_start"]), True)
    problem.add_goal(item_at(item, points[f"{name}_goal"]))

print(problem)

with OneshotPlanner(name="pcop") as planner:
    result = planner.solve(problem)
if result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
    raise SystemExit(f"No plan found: {result.status}")
plan = result.plan
print(plan)

# Grounding: the distinct unbound variables among the actions' parameters.
graph = plan._graph
unbound = {a.variable() for step in graph.nodes
           for a in step.actual_parameters if a.is_variable_exp()}
# Parallelism: actions on the longest ordering chain, counting every edge
# (conditional ones included) and only unconditional ones.
unconditional = graph.edge_subgraph(
    [(u, v) for u, v, c in graph.edges(data="condition") if c is None or c is True])
print(f"actions: {graph.number_of_nodes()}")
print(f"unbound variables: {len(unbound)} ({', '.join(sorted(v.name for v in unbound))})")
print(f"longest chain: {nx.dag_longest_path_length(graph) + 1} actions "
      f"({nx.dag_longest_path_length(unconditional) + 1 if unconditional else 1} unconditional)")
