(training-rollout-patterns)=

# Collection Patterns

Copy-paste commands for common rollout generation scenarios.

This reference provides proven patterns with complete commands, expected outputs, and post-processing scripts—organized by training objective, infrastructure, and scale.

---

## Topics

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Training Preparation
:link: training-rollout-patterns-training-prep
:link-type: ref
SFT demonstrations, DPO preference pairs, RL training buffers, and evaluation benchmarks.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Infrastructure
:link: training-rollout-patterns-infrastructure
:link-type: ref
Local vLLM, hosted APIs, distributed generation, and cost-optimized cloud patterns.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Development
:link: training-rollout-patterns-development
:link-type: ref
Quick debugging, parameter sweeps, behavioral exploration, and verification testing.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Scale
:link: training-rollout-patterns-scale
:link-type: ref
Resume interrupted runs, million-scale generation, continuous collection, and multi-dataset workflows.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Quick Command Reference
:link: training-rollout-patterns-quick-reference
:link-type: ref
Essential commands and parameters at a glance.
:::

::::

---

## Pattern Categories

```{list-table}
:header-rows: 1
:widths: 30 50 20

* - Category
  - Patterns
  - Jump To
* - **Training Preparation**
  - SFT demonstrations, DPO pairs, RL buffers, evaluation
  - {ref}`training-rollout-patterns-training-prep`
* - **Infrastructure**
  - Local vLLM, hosted API, distributed, cost-optimized
  - {ref}`training-rollout-patterns-infrastructure`
* - **Development**
  - Debug, parameter sweep, exploration, verification testing
  - {ref}`training-rollout-patterns-development`
* - **Scale**
  - Incremental/resume, million-scale, continuous, multi-dataset
  - {ref}`training-rollout-patterns-scale`
```

:::{toctree}
:maxdepth: 1
:hidden:

Training Preparation <training-preparation>
Infrastructure <infrastructure>
Development <development>
Scale <scale>
Quick Reference <quick-reference>
:::

