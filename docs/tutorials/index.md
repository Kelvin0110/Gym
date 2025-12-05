(tutorials-index)=

# NeMo Gym Tutorials

Build, train, and deploy AI agents with NeMo Gym through hands-on guided experiences.

:::{tip}
**New to NeMo Gym?** Begin with the {doc}`Get Started <../get-started/index>` section for a guided tutorial from installation through your first verified agent. Return here afterward to learn about advanced topics like additional rollout collection methods and training data generation. You can find the project repository on [GitHub](https://github.com/NVIDIA-NeMo/Gym).
:::
---

## Integrate a Training Framework

Connect Gym to popular training frameworks for end-to-end model improvement.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`hubot;1.5em;sd-mr-1` Train with TRL
:link: integrate-training-frameworks/train-with-trl
:link-type: doc
Use Hugging Face's TRL library for SFT, DPO, or GRPO training.
+++
{bdg-primary}`recommended` {bdg-secondary}`hugging-face`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: integrate-training-frameworks/train-with-nemo-rl
:link-type: doc
Use NVIDIA's NeMo RL for distributed on-policy training with NeMo 2.0 models.
+++
{bdg-secondary}`nvidia` {bdg-secondary}`multi-node`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Train with VeRL
:link: integrate-training-frameworks/train-with-verl
:link-type: doc
Use VeRL's Ray-based distributed training with flexible backend support.
+++
{bdg-secondary}`ray` {bdg-secondary}`multi-backend`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Prepare Offline Training Data
:link: integrate-training-frameworks/offline-training-w-rollouts
:link-type: doc
Transform rollouts into SFT or DPO datasets for offline training.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::

---

## Create Resource Servers

Build custom resource servers with tools, verification logic, and domain-specific functionality.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Your First Resource Server
:link: creating-resource-server
:link-type: doc
Initialize, implement, and test a resource server from scratch.
+++
{bdg-primary}`beginner` {bdg-secondary}`30 min`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Simple Tool Calling
:link: resource-servers/simple-tool-calling
:link-type: doc
Single-step tool interactions with deterministic verification.
+++
{bdg-secondary}`pattern`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Interactions
:link: resource-servers/multi-step-interactions
:link-type: doc
Orchestrate multiple tool calls for complex tasks.
+++
{bdg-secondary}`pattern`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Stateful Sessions
:link: resource-servers/stateful-sessions
:link-type: doc
Manage session state across interactions.
+++
{bdg-secondary}`pattern`
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-Judge
:link: resource-servers/llm-as-judge
:link-type: doc
Use LLMs to verify open-ended outputs.
+++
{bdg-secondary}`pattern`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Execution
:link: resource-servers/code-execution
:link-type: doc
Execute and verify generated code safely.
+++
{bdg-secondary}`pattern`
:::

::::
