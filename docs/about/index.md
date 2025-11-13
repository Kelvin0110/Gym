---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

NeMo Gym generates training data for reinforcement learning by capturing how AI agents interact with tools and environments.

## What It Does

- **Captures agent interactions**: Records what agents try, what works, and how well they perform
- **Transforms to training data**: Converts interaction records into RL-ready datasets
- **Integrates with RL frameworks**: Seamlessly hands off data to VeRL, NeMo-RL, or OpenRLHF

**Use it for**: Training agentic models that reason with tools, interact with environments, and improve through feedback.

## Core Components

Three components work together to generate and evaluate agent interactions:

**Models**  
LLM inference endpoints (OpenAI-compatible or vLLM). Handle single-turn text generation and tool-calling decisions.

**Resources**  
Provide tools (functions agents call) + verifiers (logic to score performance). Examples: math environments, code sandboxes, web search.

**Agents**  
Orchestrate multi-turn interactions between models and resources. Handle conversation flow, tool routing, and response formatting.
