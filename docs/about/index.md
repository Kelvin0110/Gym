---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

**NeMo Gym is a framework for building RL training environments.** It provides the infrastructure for creating high-quality training data through agentic model interactions.

**What it's for**: Creating data for reinforcement learning training workflows

**What you can use it for**: Training agentic models that reason with tools, interact with environments, and improve through feedback

NeMo Gym captures complete records of how AI agents interact with tools and environments—what they try, what works, and how well they perform—then transforms these records into training data for reinforcement learning workflows using the framework of your choice.

By offering unified interfaces to heterogeneous RL environments and seamless data handoff to popular RL training frameworks (VeRL, NeMo-RL, OpenRLHF), NeMo Gym lets you focus on research and model improvement rather than infrastructure and orchestration.

## Who Should Use NeMo Gym?

**Coming from SFT?** NeMo Gym extends your synthetic data generation (SDG) workflows to reinforcement learning. What you call "filtering for quality," we call "verification and reward assignment." What you call "generated examples," we call "rollouts." {doc}`See terminology mapping → <sft-to-rl-terminology>`

**Already doing RL?** You'll recognize rollout collection and reward functions. We align with standard RL terminology while adding agent-specific orchestration patterns. {doc}`See our glossary → <glossary>`

**Training agentic models?** You're ready to build. Jump straight to {doc}`your first agent <../get-started/first-agent>` or browse {doc}`tutorials <../tutorials/index>` for advanced patterns.

**New to post-training?** Start with {doc}`core concepts <concepts/index>` to understand RL fundamentals, then follow the {doc}`getting started guide <../get-started/setup-installation>` step by step.

### Quick Terminology Reference

If you're coming from SFT or other ML training approaches, here's how familiar concepts map to NeMo Gym:

| Your Background | You Know | NeMo Gym Equivalent | What It Means |
|-----------------|----------|---------------------|---------------|
| **SFT** | Synthetic data generation (SDG) | Rollout collection | Generating training examples at scale |
| **SFT** | Quality filtering | Verification + reward assignment | Scoring outputs to identify good examples |
| **SFT** | Prompt templates | Agent system prompts | Instructions that guide model behavior |
| **SFT** | Training examples | Rollouts or demonstrations | Complete interaction sequences |
| **RL (general)** | Episode | Rollout | One complete agent interaction |
| **RL (general)** | Reward function | Verifier | Logic that assigns reward signals |
| **RL (general)** | Environment | Resource Server | Tools + verification combined |
| **RL (general)** | Policy | Policy Model | The LLM being trained |

{doc}`Full glossary with all terms → <glossary>`

## Core Components

NeMo Gym organizes around three core abstractions that work together to generate and evaluate agent interactions:

* **Models**: LLM inference endpoints that generate text and make tool-calling decisions. Models are stateless and handle single-turn generation. Configure using OpenAI-compatible APIs or local vLLM servers.

* **Resources**: Servers that provide both tools (functions agents can call) and verifiers (logic to evaluate agent performance and assign reward signals). Examples include mathematical reasoning environments, code execution sandboxes, web search tools, and custom verification logic.

* **Agents**: Orchestration layers that connect models to resources, handle multi-turn conversations, route tool calls, and format responses consistently. Agents coordinate the interaction loop and can be extended with custom logic.

These components communicate via HTTP APIs and can run as separate services, enabling flexible deployment and scaling.
