(tutorials-resource-servers)=

# Create Resource Servers

Build custom resource servers that provide tools, verification logic, and domain-specific functionality for your AI agents.

---

## Getting Started

New to resource servers? Start here to understand the fundamentals.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Your First Resource Server
:link: /tutorials/creating-resource-server
:link-type: doc
Initialize a resource server, implement tools and verification, and test end-to-end.
+++
{bdg-primary}`beginner` {bdg-secondary}`30 min`
:::

::::

---

## Patterns and Recipes

Learn common patterns through concrete examples from the codebase.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Simple Tool Calling
:link: simple-tool-calling
:link-type: doc
Single-step tool interactions with deterministic verification.
+++
{bdg-secondary}`example_simple_weather`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Interactions
:link: multi-step-interactions
:link-type: doc
Orchestrate multiple tool calls to complete complex tasks.
+++
{bdg-secondary}`example_multi_step`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Stateful Sessions
:link: stateful-sessions
:link-type: doc
Manage session state across multiple interactions.
+++
{bdg-secondary}`example_stateful_counter`
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-Judge Verification
:link: llm-as-judge
:link-type: doc
Use language models to verify open-ended outputs.
+++
{bdg-secondary}`math_with_judge`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Execution
:link: code-execution
:link-type: doc
Execute and verify generated code safely.
+++
{bdg-secondary}`code_gen`
:::

::::

---

```{toctree}
:maxdepth: 1
:hidden:

simple-tool-calling
multi-step-interactions
stateful-sessions
llm-as-judge
code-execution
```

