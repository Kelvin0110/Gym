(training-data-quality)=

# Data Quality for Training

Validate training data quality using NeMo Gym's automatic metrics and prepare data for RL frameworks.

Generated rollouts need quality validation before training—learn to quickly assess your data.

## When You Need This

Use this section when you need to:

* **Validate rollout quality** - Check if data is ready for training
* **Interpret automatic metrics** - Understand NeMo Gym's metric aggregation
* **Assess quality by RL algorithm** - Ensure data meets PPO, DPO, or other algorithm needs
* **Catch verification issues** - Identify broken verification before wasting training compute

:::{note}
**NeMo Gym's role**: Generate high-quality rollouts at scale. Your RL framework (VeRL, NeMo-RL, OpenRLHF, TRL) handles training. This section covers the validation handoff between them.
:::

---

## Quality Validation Guide

::::{grid} 1
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Quality Metrics
:link: quality-metrics
:link-type: doc

**How-to guide** for validating rollout quality using NeMo Gym's automatic metrics and quick checks before passing to RL frameworks.
+++
{bdg-secondary}`how-to` {bdg-secondary}`metrics` {bdg-secondary}`validation`
:::

::::

---

## Quality Pipeline

Data quality validation in the training pipeline:

```
Generated Rollouts (NeMo Gym)
    ↓
[Validate Quality]  ← NeMo Gym automatic metrics
    ↓
[Format Check]      ← ng_prepare_data
    ↓
RL Framework        → VeRL, NeMo-RL, OpenRLHF, TRL
```

**Previous**: {doc}`../rollout-collection/index` for generation  
**Next**: {doc}`../datasets/index` for format validation  
**Then**: {doc}`../integration/index` for RL framework training

---

## Quick Quality Checks

Use NeMo Gym's built-in features to validate before training:

### Automatic Metric Aggregation

```bash
ng_collect_rollouts +input_jsonl_fpath=tasks.jsonl +output_jsonl_fpath=rollouts.jsonl

# Automatically displays after collection:
# {
#   "reward": 0.73,
#   "accuracy": 0.68,
#   "avg_tool_calls": 2.1
# }
```

### Quick Statistics

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl
```

### Smoke Tests

See {doc}`../rollout-collection/optimize-for-training/production-scale` for quick quality checks during collection.

---

## Quality by RL Algorithm

Different algorithms have different data requirements:

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Algorithm
  - Data Requirement
  - Expected Quality Signal
* - **PPO**
  - Diverse quality range
  - Reward spread 0.3-0.9, not clustered
* - **DPO**
  - Preference pairs with gaps
  - Both high (>0.7) and low (<0.5) rewards
* - **SFT**
  - High-quality demonstrations
  - Mean reward ≥ 0.8, success rate ≥ 80%
```

See {doc}`quality-metrics` for validation by algorithm type.

---

## Custom Filtering and Balancing

**NeMo Gym does not provide built-in filtering or balancing utilities.**

If your RL framework or training objectives require filtering or balancing:

1. **Implement based on your RL framework's requirements**
   - Each framework (VeRL, NeMo-RL, OpenRLHF, TRL) has different data preferences
   - See your framework's documentation for recommended preprocessing

2. **Use quality metrics to guide custom filtering**
   - NeMo Gym's automatic metrics help identify thresholds
   - Apply filtering based on your training objectives

3. **Consider framework-native preprocessing**
   - Many RL frameworks include data filtering and balancing
   - Leverage framework capabilities rather than custom pipelines

**Example filtering patterns** are shown in {doc}`../tutorials/offline-training-w-rollouts` tutorial.

---

## Related Topics

### Data Pipeline

* {doc}`../rollout-collection/index` - Generate rollouts (before quality validation)
* {doc}`../verification/index` - Design rewards that enable quality filtering
* {doc}`../datasets/index` - Format and validate data for training

### Analysis Tools

* `ng_collect_rollouts` - Automatic metric aggregation (built-in)
* `ng_prepare_data` - Dataset format validation (see {doc}`../datasets/validate-format`)
* `scripts/print_aggregate_results.py` - Quick metric summaries (built-in)

---

## Next Steps

:::{button-ref} quality-metrics
:color: primary
:outline:
:ref-type: doc

Start with Quality Metrics →
:::

:::{tip}
**Unsure about quality?** Start with {doc}`quality-metrics` to understand your data distribution, then proceed to {doc}`../datasets/validate-format` for format validation before training.
:::

```{toctree}
:hidden:
:maxdepth: 1

quality-metrics
```
