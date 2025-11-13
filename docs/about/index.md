---
orphan: true
---

(about-overview)=
# About NVIDIA NeMo Gym

NeMo Gym provides infrastructure to generate and evaluate agent behavior at scale. It captures complete records of how AI agents interact with tools and environments—what they try, what works, and how well they perform—then transforms these records into training data for reinforcement learning workflows using the framework of your choice.

By offering unified interfaces to heterogeneous RL environments and seamless data handoff to popular RL training frameworks (VeRL, NeMo-RL, OpenRLHF), NeMo Gym lets you focus on research and model improvement rather than infrastructure and orchestration.

## Core Components

NeMo Gym organizes around three core abstractions that work together to generate and evaluate agent interactions:

* **Models**: LLM inference endpoints that generate text and make tool-calling decisions. Models are stateless and handle single-turn generation. Configure using OpenAI-compatible APIs or local vLLM servers.

* **Resources**: Servers that provide both tools (functions agents can call) and verifiers (logic to evaluate agent performance and assign reward signals). Examples include mathematical reasoning environments, code execution sandboxes, web search tools, and custom verification logic.

* **Agents**: Orchestration layers that connect models to resources, handle multi-turn conversations, route tool calls, and format responses consistently. Agents coordinate the interaction loop and can be extended with custom logic.

These components communicate via HTTP APIs and can run as separate services, enabling flexible deployment and scaling.
