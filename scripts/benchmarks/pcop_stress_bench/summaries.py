"""Paper-friendly summaries for the PCOP stress benchmarks.

These are adapted from the COAST Blocks, Kitchen, and Rover domains, but with
motion planning stripped away so the benchmark stresses symbolic plan structure.
"""

FAMILY_SUMMARIES = {
    'blocks_clearance': (
        'Multi-agent symbolic adaptation of the COAST Blocks domain. Two manipulators '
        'must clear obstructing blocks to free a target block and its goal cell. The '
        'benchmark isolates obstruction clearing, non-monotonic rearrangement, and '
        'agent interference without geometric refinement.'
    ),
    'blocks_handover': (
        'Coordination-focused Blocks variant in which no single agent can complete the '
        'task alone. Target blocks must cross a shared transfer region, stressing '
        'partial-order flexibility around handoff timing and shared bottlenecks.'
    ),
    'kitchen_pipeline': (
        'Multi-agent symbolic adaptation of the COAST Kitchen domain. Two robots '
        'must clean, transfer, cook, and serve several items using shared sink, stove, '
        'and pass-table resources. The benchmark preserves long causal chains and '
        'repeated transfers while removing motion planning.'
    ),
    'kitchen_buffer_trap': (
        'Order-sensitive Kitchen variant with one-slot shared resources. Poor action '
        'orderings fill the sink, stove, or pass buffer and force extra rearrangement, '
        'making this benchmark a stress test for sequence-sensitive symbolic planning.'
    ),
    'rover_relay': (
        'Multi-agent symbolic adaptation of the COAST Rover domain. Multiple rovers '
        'must gather data in disconnected regions, rendezvous at relay nodes, and '
        'uplink results to a lander. The benchmark isolates relay coordination, task '
        'allocation, and synchronization points.'
    ),
}

PROBLEM_NOTES = {
    'blocks_clearance_small': 'Two obstacle-clearing subgoals are nearly independent, so a good partial-order plan should keep them weakly ordered until the target move becomes necessary.',
    'blocks_clearance_medium': 'Adds an extra blocker and a tighter parking layout, increasing interference and threat density in pick/place reasoning.',
    'blocks_handover_small': 'Forces a single transfer through a shared cell because each manipulator can only reach one side of the workspace.',
    'blocks_handover_medium': 'Two target blocks must cross the same transfer bottleneck, increasing synchronization pressure and handoff competition.',
    'kitchen_pipeline_small': 'Two chefs clean and cook two items through a one-slot pass table, creating reusable handoff structure with limited shared resources.',
    'kitchen_pipeline_medium': 'Extends the kitchen pipeline to three items, increasing long-horizon causal chaining and pass/stove contention.',
    'kitchen_buffer_trap_small': 'A one-slot sink, stove, and pass table create a small but order-sensitive scheduling problem with shared buffers.',
    'kitchen_buffer_trap_medium': 'Three items and single-capacity shared resources produce many near-equivalent prefixes but few good global orderings.',
    'rover_relay_small': 'One rover must relay sampled data through a center node while another both observes and uplinks from the base side.',
    'rover_relay_medium': 'Three rovers gather data from left, north, and right regions and coordinate at a shared hub before uplink.',
    'rover_relay_hard': 'Adds one more disconnected objective and keeps a sparse relay topology, increasing both allocation ambiguity and synchronization load.',
}
