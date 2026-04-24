from stress_problems import (
    make_blocks_clearance_small,
    make_blocks_clearance_medium,
    make_blocks_handover_small,
    make_blocks_handover_medium,
    make_kitchen_pipeline_small,
    make_kitchen_pipeline_medium,
    make_kitchen_buffer_trap_small,
    make_kitchen_buffer_trap_medium,
    make_rover_relay_small,
    make_rover_relay_medium,
    make_rover_relay_hard,
)
from summaries import FAMILY_SUMMARIES, PROBLEM_NOTES


PCOP_STRESS_PROBLEMS = {
    'blocks_clearance_small': {
        'builder': make_blocks_clearance_small,
        'family': 'blocks_clearance',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place'},
        'summary': PROBLEM_NOTES['blocks_clearance_small'],
    },
    'blocks_clearance_medium': {
        'builder': make_blocks_clearance_medium,
        'family': 'blocks_clearance',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place'},
        'summary': PROBLEM_NOTES['blocks_clearance_medium'],
    },
    'blocks_handover_small': {
        'builder': make_blocks_handover_small,
        'family': 'blocks_handover',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place'},
        'summary': PROBLEM_NOTES['blocks_handover_small'],
    },
    'blocks_handover_medium': {
        'builder': make_blocks_handover_medium,
        'family': 'blocks_handover',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place'},
        'summary': PROBLEM_NOTES['blocks_handover_medium'],
    },
    'kitchen_pipeline_small': {
        'builder': make_kitchen_pipeline_small,
        'family': 'kitchen_pipeline',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place', 'clean_item', 'cook_item'},
        'summary': PROBLEM_NOTES['kitchen_pipeline_small'],
    },
    'kitchen_pipeline_medium': {
        'builder': make_kitchen_pipeline_medium,
        'family': 'kitchen_pipeline',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place', 'clean_item', 'cook_item'},
        'summary': PROBLEM_NOTES['kitchen_pipeline_medium'],
    },
    'kitchen_buffer_trap_small': {
        'builder': make_kitchen_buffer_trap_small,
        'family': 'kitchen_buffer_trap',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place', 'clean_item', 'cook_item'},
        'summary': PROBLEM_NOTES['kitchen_buffer_trap_small'],
    },
    'kitchen_buffer_trap_medium': {
        'builder': make_kitchen_buffer_trap_medium,
        'family': 'kitchen_buffer_trap',
        'sync_actions': set(),
        'resource_actions': {'pick', 'place', 'clean_item', 'cook_item'},
        'summary': PROBLEM_NOTES['kitchen_buffer_trap_medium'],
    },
    'rover_relay_small': {
        'builder': make_rover_relay_small,
        'family': 'rover_relay',
        'sync_actions': {'relay'},
        'resource_actions': {'navigate', 'sample', 'observe', 'relay', 'uplink_data'},
        'summary': PROBLEM_NOTES['rover_relay_small'],
    },
    'rover_relay_medium': {
        'builder': make_rover_relay_medium,
        'family': 'rover_relay',
        'sync_actions': {'relay'},
        'resource_actions': {'navigate', 'sample', 'observe', 'relay', 'uplink_data'},
        'summary': PROBLEM_NOTES['rover_relay_medium'],
    },
    'rover_relay_hard': {
        'builder': make_rover_relay_hard,
        'family': 'rover_relay',
        'sync_actions': {'relay'},
        'resource_actions': {'navigate', 'sample', 'observe', 'relay', 'uplink_data'},
        'summary': PROBLEM_NOTES['rover_relay_hard'],
    },
}


FAMILY_SUMMARY_ROWS = {
    family: {'family': family, 'summary': summary}
    for family, summary in FAMILY_SUMMARIES.items()
}
