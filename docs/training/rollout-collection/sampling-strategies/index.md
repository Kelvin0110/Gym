(training-rollout-sampling)=

# Sampling Strategies

Configure temperature, diversity, and sampling parameters to match your training objective.

Different training algorithms need different data characteristics—learn how to tune rollout generation for SFT, DPO, RL, and evaluation.

---

## Quick Strategy Guide

```{list-table}
:header-rows: 1
:widths: 20 15 15 15 35

* - Training Type
  - Temperature
  - Num Repeats
  - Parallelism
  - Goal
* - **SFT**
  - Low
  - 1
  - High
  - Consistent demonstrations at scale
* - **DPO**
  - Higher
  - 3-4
  - Medium
  - Diverse preference pairs for comparison
* - **RL**
  - Moderate
  - 1
  - Medium-High
  - Balance exploration and exploitation
* - **Evaluation**
  - Very low
  - 1
  - Low
  - Reproducible, deterministic results
* - **Research**
  - High
  - 5+
  - Medium
  - Maximum diversity for discovery
```

---

## Guides and References

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Understanding Parameters
:link: parameters
:link-type: doc

**Explanation** of temperature, top_p, and num_repeats—what they control and how they affect rollout characteristics.
+++
{bdg-secondary}`explanation` {bdg-secondary}`fundamentals`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` SFT Sampling Strategy
:link: sft
:link-type: doc

**How-to guide** for supervised fine-tuning: consistent, high-quality demonstrations at scale.
+++
{bdg-secondary}`how-to` {bdg-secondary}`sft` {bdg-secondary}`consistency`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` DPO Sampling Strategy
:link: dpo
:link-type: doc

**How-to guide** for Direct Preference Optimization: generating diverse preference pairs with quality differences.
+++
{bdg-secondary}`how-to` {bdg-secondary}`dpo` {bdg-secondary}`diversity`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` RL Sampling Strategy
:link: rl
:link-type: doc

**How-to guide** for reinforcement learning: balancing exploration and exploitation with iterative collection.
+++
{bdg-secondary}`how-to` {bdg-secondary}`rl` {bdg-secondary}`exploration`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Evaluation Sampling Strategy
:link: evaluation
:link-type: doc

**How-to guide** for benchmarking: reproducible, deterministic evaluation with minimal variance.
+++
{bdg-secondary}`how-to` {bdg-secondary}`evaluation` {bdg-secondary}`reproducibility`
:::

:::{grid-item-card} {octicon}`telescope;1.5em;sd-mr-1` Research Sampling Strategy
:link: research
:link-type: doc

**How-to guide** for behavioral exploration: discovering capabilities, failure modes, and edge cases.
+++
{bdg-secondary}`how-to` {bdg-secondary}`research` {bdg-secondary}`analysis`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Measuring Success
:link: validation
:link-type: doc

**Reference** for validating your sampling strategy through reward distributions and diversity metrics.
+++
{bdg-secondary}`reference` {bdg-secondary}`validation` {bdg-secondary}`metrics`
:::

::::

---

## Strategy Selection Flow

Use this flowchart to determine which sampling strategy matches your training goal.

~~~{mermaid}
flowchart TD
    Start[What is your goal?] --> Training{Training or<br/>Evaluation?}
    
    Training -->|Training| Algorithm{Which algorithm?}
    Training -->|Evaluation| Eval[Use Evaluation Strategy<br/>deterministic, repeatable]
    
    Algorithm -->|SFT| SFT[Use SFT Strategy<br/>low temp, single sample]
    Algorithm -->|DPO| DPO[Use DPO Strategy<br/>higher temp, multiple repeats]
    Algorithm -->|RL| RL[Use RL Strategy<br/>moderate temp, iterative]
    Algorithm -->|Research/Debug| Research[Use Research Strategy<br/>high temp, many repeats]
    
    SFT --> Filter[Filter for high rewards]
    DPO --> Pair[Create preference pairs]
    RL --> Iterate[Iterative collection<br/>+ policy updates]
    Eval --> Compare[Compare models<br/>fixed seed]
    Research --> Analyze[Analyze variance<br/>+ failure modes]
~~~

---

## Quick Command Templates

Copy and customize these templates for each training type. Configure `temperature`, `num_repeats`, and `num_samples_in_parallel` based on your specific needs.

::::{tab-set}

:::{tab-item} SFT

```bash
ng_collect_rollouts \
    +agent_name=<agent_name> \
    +input_jsonl_fpath=<input_file> \
    +output_jsonl_fpath=<output_file> \
    +responses_create_params.temperature=<temperature> \
    +num_samples_in_parallel=<parallelism>
```

**Typical settings**: Low temperature (consistency), high parallelism (scale)

:::

:::{tab-item} DPO

```bash
ng_collect_rollouts \
    +agent_name=<agent_name> \
    +input_jsonl_fpath=<input_file> \
    +output_jsonl_fpath=<output_file> \
    +responses_create_params.temperature=<temperature> \
    +num_repeats=<num_repeats> \
    +num_samples_in_parallel=<parallelism>
```

**Typical settings**: Higher temperature (diversity), multiple repeats (3-4), moderate parallelism

:::

:::{tab-item} RL

```bash
ng_collect_rollouts \
    +agent_name=<agent_name> \
    +input_jsonl_fpath=<input_file> \
    +output_jsonl_fpath=<output_file> \
    +responses_create_params.temperature=<temperature> \
    +num_samples_in_parallel=<parallelism>
```

**Typical settings**: Moderate temperature (exploration/exploitation balance), medium-high parallelism

:::

:::{tab-item} Evaluation

```bash
ng_collect_rollouts \
    +agent_name=<agent_name> \
    +input_jsonl_fpath=<input_file> \
    +output_jsonl_fpath=<output_file> \
    +responses_create_params.temperature=<temperature> \
    +responses_create_params.seed=<seed> \
    +num_samples_in_parallel=<parallelism>
```

**Typical settings**: Very low temperature (determinism), fixed seed (reproducibility), low parallelism

:::

::::

## Next Steps

Start with **{doc}`parameters`** to understand the fundamentals, then choose the strategy guide that matches your training objective.

:::{button-ref} parameters
:color: primary
:outline:
:ref-type: doc

Start with Understanding Parameters →
:::

```{toctree}
:hidden:
:maxdepth: 1

Sampling Parameters <parameters>
SFT <sft>
DPO <dpo>
RL <rl>
Evaluation <evaluation>
Research <research>
Validation <validation>
```
