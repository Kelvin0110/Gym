(training-rollout-optimize)=

# Optimize for Training

Maximize rollout generation throughput to build large-scale training datasets efficiently.

:::{card}

**Task**: Generate thousands of rollouts per hour by identifying bottlenecks and tuning your infrastructure for maximum throughput.

^^^

**This guide shows you how to**:

1. Identify your bottleneck (model, verification, or network)
2. Tune parallelism for your infrastructure
3. Optimize model server configuration
4. Monitor and scale to production

:::

---

## Before You Start

Ensure you have these prerequisites before optimizing your rollout collection:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Details
* - **Get Started completed**
  - Complete {doc}`../../../get-started/collecting-rollouts` first
* - **Servers running**
  - Agent and model servers collecting rollouts successfully
* - **Infrastructure access**
  - Ability to configure parallelism and model server settings
* - **Basic metrics**
  - Know your current throughput (samples/sec or time per 100 samples)
* - **Infrastructure type**
  - Local GPU (vLLM/TensorRT-LLM) OR hosted API (OpenAI/Azure) OR NVIDIA NIM
```

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Optimization Workflow

Follow this workflow to systematically improve throughput:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` 1. Identify Bottleneck
:link: identify-bottleneck
:link-type: doc

Diagnose what's limiting your throughput: model inference, verification complexity, or network/API limits.

**Start here** if you don't know what's slowing you down.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` 2. Tune Parallelism
:link: tune-parallelism
:link-type: doc

Optimize `num_samples_in_parallel` to maximize concurrent requests without overwhelming infrastructure.

**Use this** after identifying your bottleneck.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` 3. Optimize Infrastructure
:link: optimize-infrastructure
:link-type: doc

Configure model servers (vLLM) and verification logic for maximum performance.

**Apply these** for model or verification bottlenecks.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` 4. Production Scale
:link: production-scale
:link-type: doc

Monitor metrics, handle interruptions, and deploy production patterns for million-scale generation.

**Use this** when ready for production workloads.
:::

::::

---

## Quick Decision Guide

Jump directly to the right section based on your current need:

```{list-table}
:header-rows: 1
:widths: 40 60

* - If You Need To...
  - Go To
* - **Diagnose why collection is slow**
  - {doc}`identify-bottleneck` - Run diagnostic tests
* - **Configure parallelism settings**
  - {doc}`tune-parallelism` - Find your optimal value
* - **Speed up vLLM inference**
  - {doc}`optimize-infrastructure` (Model Server section)
* - **Fix slow verification**
  - {doc}`optimize-infrastructure` (Verification section)
* - **Scale to millions of rollouts**
  - {doc}`production-scale` - Production patterns
* - **Debug OOM or timeout errors**
  - {doc}`production-scale` (Troubleshooting section)
```

---

## Quick Reference

### Parallelism by Infrastructure

| Infrastructure | Recommended `num_samples_in_parallel` |
|----------------|---------------------------------------|
| Local vLLM (1 GPU) | 10-15 |
| Local vLLM (4 GPU) | 20-40 |
| OpenAI API | 5-10 |
| Azure OpenAI Scale | 8-15 |
| Self-hosted cluster | 30-100 |

### Optimization Checklist

- [ ] Identified bottleneck (model, verification, or network)
- [ ] Tuned `num_samples_in_parallel` for infrastructure
- [ ] Reduced `max_output_tokens` to minimum needed
- [ ] Optimized model server configuration (if local)
- [ ] Cached expensive verification computations
- [ ] Benchmarked throughput (samples/sec)
- [ ] Tested stability with larger dataset
- [ ] Set up monitoring for production runs

---

## Next Steps

After optimizing throughput:

**Tune data characteristics**: {doc}`../sampling-strategies`  
Learn how to configure temperature and sampling for different training objectives.

**See complete patterns**: {doc}`../collection-patterns`  
Reference guide with copy-paste commands for common scenarios.

**Filter and curate**: {doc}`../../data-quality/index`  
Improve training data quality through systematic filtering.

---

```{toctree}
:maxdepth: 1
:hidden:

identify-bottleneck
tune-parallelism
optimize-infrastructure
production-scale
```

