---
orphan: true
---

(about-concepts)=
# Understanding Concepts for {{product_name}}

NeMo Gym concepts explain the mental model behind building reliable agent systems. This section provides essential terminology and explains how teams capture interaction data for reinforcement learning workflows.

---

## How to Navigate This Section

- Start with the **Glossary** to learn reinforcement learning and agent training terminology.
- Read **Rollout Collection Fundamentals** to understand how interaction data is captured and structured for training.
- Pair these concepts with hands-on tutorials when you are ready to practice tasks such as assembling interaction datasets or scoring agent behavior.

---

## Concept Highlights

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` RL Terms Glossary
:link: glossary
:link-type: doc
Essential terminology for agent training, reinforcement learning workflows, and NeMo Gym architecture organized by category for easy reference.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection Fundamentals
:link: concepts-rc-fundamentals
:link-type: ref
Learn why complete interaction transcripts matter for reinforcement learning, how they enable evaluation, and how collection orchestrators stream results to JSONL datasets.
:::

::::

---

## Continue Learning

- Apply these concepts in practice by stepping through {doc}`../../get-started/first-agent` to create your first agent interaction.
- Move to high-volume data generation with {doc}`../../get-started/collecting-rollouts` once the fundamentals are familiar.
- Reference the complete {doc}`../features` catalog to understand all available capabilities.

---

```{toctree}
:hidden:
:maxdepth: 1

RL Terms Glossary <glossary>
Rollout Collection Fundamentals <rollout-collection-fundamentals>
```
