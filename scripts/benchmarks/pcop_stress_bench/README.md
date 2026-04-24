# PCOP stress benchmarks

This folder contains a PCOP-focused stress suite inspired by the symbolic structure
of the COAST Blocks, Kitchen, and Rover domains.

## Files

- `benchmark_pcop_stress.py` — main benchmark runner
- `stress_metrics.py` — extra plan-structure metrics
- `stress_registry.py` — benchmark registry and per-problem metadata
- `summaries.py` — paper-friendly summaries for each family/problem
- `stress_problems/blocks.py` — Blocks-style stress tests
- `stress_problems/kitchen.py` — Kitchen-style stress tests
- `stress_problems/rover.py` — Rover-style stress tests

## Example usage

```bash
python benchmark_pcop_stress.py
python benchmark_pcop_stress.py --timeout 60
python benchmark_pcop_stress.py --only blocks_clearance_small rover_relay_small
```

## Reported metrics

In addition to runtime and solve status, the benchmark script records:

- `action_count` — number of action instances in the returned plan
- `direct_orderings` — number of explicit ordering edges in the partial-order plan
- `comparable_pairs` — action pairs ordered transitively
- `unordered_pairs` — action pairs left flexible / unordered
- `flexibility_ratio` — unordered pair fraction
- `unique_agent_count` — how many agent/rover objects appear in the plan
- `sync_action_count` — number of relay / synchronization actions
- `resource_action_count` — number of resource-touching actions
- `binding_var_count` and related counts — partial-binding statistics when available

These metrics are meant to help evaluate whether PCOP is exploiting partial-order
flexibility rather than only comparing raw runtime.
