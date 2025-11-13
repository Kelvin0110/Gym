---
orphan: true
---

(sft-to-rl-terminology)=
# SFT to RL Terminology Guide

If you're coming from supervised fine-tuning (SFT) workflows, this guide maps your existing knowledge to NeMo Gym's reinforcement learning concepts. The core ideas are similar—you're still generating training data at scale—but the terminology and approach differ.

## Core Concept Mapping

### Data Generation: SDG → Rollout Collection

**What you know (SFT)**: Synthetic Data Generation (SDG)
- Generate many examples using prompts
- Run inference on your model or a teacher model
- Collect the outputs as training data

**NeMo Gym equivalent**: Rollout Collection
- Generate many interactions using task prompts
- Run agents that use tools and environments
- Collect complete interaction sequences (rollouts) as training data

**Key difference**: In SFT, you generate isolated responses. In RL, you generate complete multi-step interactions that include tool usage, intermediate reasoning, and final outcomes.

```python
# SFT style: single prompt → single response
prompt = "Solve: 2 + 2 = ?"
response = model.generate(prompt)  # "4"
training_example = (prompt, response)

# NeMo Gym style: task → multi-step rollout
task = "What is the weather in Seattle?"
rollout = agent.execute(task)
# Rollout includes:
# - User request
# - Agent reasoning ("I need to call weather API")
# - Tool call (get_weather(city="Seattle"))
# - Tool result ({"temp": 65, "condition": "cloudy"})
# - Final response ("It's 65°F and cloudy in Seattle")
# - Verification score (1.0 = correct answer)
```

---

### Quality Control: Filtering → Verification + Rewards

**What you know (SFT)**: Quality Filtering
- Filter generated examples to keep only good ones
- Use heuristics, rule-based checks, or reward models
- Binary decision: keep or discard

**NeMo Gym equivalent**: Verification + Reward Assignment
- Score every rollout with a continuous reward signal (0.0–1.0)
- Use verifiers that check correctness, efficiency, or quality
- All rollouts are kept—the reward score tells the training algorithm which behaviors to reinforce

**Key difference**: Instead of binary filtering, you assign continuous scores. This enables RL algorithms to learn from both successes and failures.

```python
# SFT style: filter binary
examples = generate_many_examples(prompts)
good_examples = [ex for ex in examples if passes_quality_check(ex)]
# Discard bad examples

# NeMo Gym style: score everything
rollouts = collect_rollouts(tasks)
for rollout in rollouts:
    rollout.reward = verifier.score(rollout)
    # Keep rollout regardless of score
    # RL algorithm learns: high reward = good, low reward = bad
```

---

### Training Data Format

**What you know (SFT)**: Demonstrations
- Pairs of (prompt, good_response)
- Model learns to mimic good responses
- All examples are "correct"

**NeMo Gym equivalent**: Scored Rollouts
- Complete interaction sequences with reward scores
- Model learns which actions lead to high rewards
- Examples include both successes and failures

**Format comparison**:

```json
// SFT demonstration format
{
  "prompt": "What is 2+2?",
  "response": "2+2 equals 4."
}

// NeMo Gym rollout format (simplified)
{
  "prompt": "What is 2+2?",
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "Let me calculate that."},
    {"role": "assistant", "tool_calls": [{"function": "calculator", "arguments": {"expression": "2+2"}}]},
    {"role": "tool", "content": "4"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ],
  "reward": 1.0,
  "metadata": {
    "steps_taken": 2,
    "tools_used": ["calculator"],
    "verified": true
  }
}
```

---

### Infrastructure: Generation Scripts → Server Architecture

**What you know (SFT)**: Simple Generation Scripts
- Script that calls model API
- Processes prompts in batch
- Writes outputs to files

**NeMo Gym equivalent**: Distributed Server Architecture
- **Model servers**: LLM inference endpoints
- **Resource servers**: Tool + verification services
- **Agent servers**: Orchestration between models and resources
- CLI commands to manage distributed services

**Key difference**: NeMo Gym uses a service-oriented architecture because agent workflows involve multiple services (models, tools, verifiers) that need to coordinate.

---

## Terminology Quick Reference

| SFT Term | NeMo Gym Term | Definition |
|----------|---------------|------------|
| Synthetic Data Generation (SDG) | Rollout Collection | Generating training examples at scale |
| Generated Example | Rollout | One complete interaction sequence |
| Prompt Template | Task Prompt + Agent System Prompt | Instructions for the agent |
| Quality Filter | Verifier | Logic that scores outputs |
| Keep/Discard | Reward Score | Continuous signal (0.0–1.0) |
| Teacher Model | Policy Model or Judge Model | Model providing outputs or evaluations |
| Demonstration | Rollout with high reward | Example of good behavior |
| Training Dataset | Rollout Dataset | Collection of scored interactions |

---

## Server Concepts (New for RL)

In SFT, you typically call a single model API. In NeMo Gym, you work with three types of servers:

### 1. Model Servers (Familiar Territory)

These are LLM inference endpoints—similar to what you use in SFT.

- **OpenAI API**: Commercial models (GPT-4, etc.)
- **vLLM**: Self-hosted open models (Llama, Mistral, etc.)
- **Azure OpenAI**: Enterprise-hosted models

**Difference from SFT**: In NeMo Gym, models follow the OpenAI Responses API format, which supports tool calling and multi-turn conversations natively.

### 2. Resource Servers (New Concept)

These provide:
- **Tools**: Functions agents can call (APIs, calculators, databases)
- **Verifiers**: Logic to score agent performance

Think of resources as "the environment" the agent interacts with.

**Examples**:
- Math resource: provides calculator tool + correctness verifier
- Web resource: provides search tool + relevance verifier
- Code resource: provides code executor + test suite verifier

### 3. Agent Servers (New Concept)

These orchestrate interactions between models and resources:
- Receive user requests
- Call the model to generate responses
- Execute tool calls against resource servers
- Format multi-turn conversations
- Return complete rollouts with verification scores

**Simple analogy**: If the model is the "brain" and resources are the "hands and eyes," the agent is the "coordinator" that connects them.

---

## Workflow Comparison

### SFT Workflow

```bash
1. Write prompts → prompts.jsonl
2. Run generation script
   for prompt in prompts:
       response = model.generate(prompt)
       examples.append((prompt, response))
3. Filter for quality
   good_examples = filter(examples)
4. Train model on good_examples
```

### NeMo Gym Workflow

```bash
1. Write task prompts → tasks.jsonl
2. Start servers (model + resources + agent)
   ng_run +config_paths=[model.yaml,resource.yaml]
3. Collect rollouts
   ng_collect_rollouts \
       +agent_name=my_agent \
       +input_jsonl_fpath=tasks.jsonl \
       +output_jsonl_fpath=rollouts.jsonl
   # Rollouts automatically include verification scores
4. Train model on scored rollouts (using RL framework)
```

**Key difference**: Steps 2–3 involve distributed services instead of a single script. This enables parallel execution, tool usage, and real-time verification.

---

## What You Need to Learn

Based on your SFT background, here's what's new:

### ✅ You Already Know

- How to generate synthetic training data at scale
- How to use model APIs (OpenAI, vLLM, etc.)
- How to format prompts and structure datasets
- How to evaluate output quality
- How training loops work (forward pass, loss, optimizer)

### 🆕 New Concepts to Learn

**High Priority** (needed immediately):
1. **Rollout structure**: Multi-step interactions vs single responses
2. **Verification and rewards**: Continuous scoring vs binary filtering
3. **Server architecture**: Model + Resources + Agent coordination
4. **Tool calling**: How agents use external functions

**Medium Priority** (helpful for advanced usage):
5. **Multi-turn orchestration**: Managing conversation state
6. **RL training integration**: How rollouts feed VeRL/NeMo-RL
7. **Resource server patterns**: Building custom verifiers

**Lower Priority** (can skip initially):
8. **Distributed deployment**: Running services across machines
9. **Advanced agent patterns**: Custom orchestration logic
10. **Performance optimization**: Batching, caching, parallelization

---

## Learning Path for SFT Users

### Step 1: Understand Rollouts (15 min)

Read {doc}`Rollout Collection Fundamentals <concepts/rollout-collection-fundamentals>` to understand:
- What a rollout contains
- How verification works
- Why multi-step matters

### Step 2: Run Your First Agent (30 min)

Follow {doc}`../get-started/setup-installation` and {doc}`../get-started/first-agent` to:
- Start model + resource + agent servers
- Interact with an agent
- See a complete rollout

### Step 3: Collect Rollouts at Scale (20 min)

Follow {doc}`../get-started/collecting-rollouts` to:
- Generate rollouts from task prompts
- Understand the JSONL output format
- Compare to your SFT data generation scripts

### Step 4: Understand Verification (20 min)

Read {doc}`../get-started/verifying-agent-results` to:
- See how reward scores are computed
- Understand verifier patterns
- Learn when verification happens

### Step 5: Use Rollouts for Training (Advanced)

Browse {doc}`../tutorials/offline-training-w-rollouts` to:
- Export rollouts to RL training frameworks
- Integrate with VeRL, NeMo-RL, or OpenRLHF
- Run your first RL training loop

---

## Common Questions from SFT Users

### "Do I need to learn RL algorithms?"

**No, not initially.** You can think of NeMo Gym as "SDG for RL." You generate scored rollouts, and the RL framework handles training. Focus on:
1. Generating good rollouts (you already know this from SFT)
2. Writing effective verifiers (similar to quality filters)
3. Understanding the server architecture

Later, if you want to tune RL hyperparameters or customize algorithms, you'll need deeper RL knowledge—but that's optional.

### "Why can't I just use my SFT scripts?"

You can use similar scripts, but NeMo Gym adds:
- **Tool calling**: Agents need to interact with external services
- **Verification**: Automatic scoring of outputs
- **Multi-step reasoning**: Agents break tasks into steps
- **Scalability**: Distributed architecture for production workloads

If your task is simple (no tools, single-turn), you could adapt SFT scripts. But for agentic workflows, NeMo Gym's architecture provides significant value.

### "What's the difference between demonstrations and rollouts?"

**Demonstrations** (SFT) = examples of correct behavior only
**Rollouts** (RL) = examples of any behavior, scored by reward

In NeMo Gym:
- High-reward rollouts can be used as demonstrations for SFT
- Mixed-reward rollouts can be used for preference learning (DPO)
- All rollouts with scores can be used for RL training

So rollouts are more general—they support SFT, DPO, and RL training.

### "Do I need to run multiple servers?"

For local development and experimentation: **Yes, but it's simple.**
- One command (`ng_run`) starts all servers
- Everything runs on your laptop/workstation
- Servers communicate via localhost

For production: **Yes, and you benefit from distribution.**
- Model servers can run on GPU machines
- Resource servers can run on CPU machines
- Agents can scale independently

The architecture is designed for scale, but simple to run locally.

---

## Next Steps

Ready to build? Here's your path:

1. **Install NeMo Gym**: {doc}`../get-started/setup-installation`
2. **Run first agent**: {doc}`../get-started/first-agent`
3. **Collect rollouts**: {doc}`../get-started/collecting-rollouts`
4. **Browse examples**: {doc}`../tutorials/index`

Need terminology help? {doc}`See full glossary → <glossary>`

---

## Reference: Full Term Mapping

Comprehensive mapping of SFT terminology to NeMo Gym equivalents:

| SFT Concept | NeMo Gym Concept | Notes |
|-------------|------------------|-------|
| Synthetic Data Generation | Rollout Collection | Generating training data |
| Prompt | Task | Input to the agent |
| Response | Rollout | Complete interaction sequence |
| Single-turn | Single-step | One model call |
| Multi-turn conversation | Multi-turn rollout | Conversation with context |
| N/A | Multi-step rollout | Agent breaks task into steps |
| Teacher model | Policy model or judge model | Depends on role |
| Student model | Policy model | Model being trained |
| Generated text | Agent response | Output from model |
| N/A | Tool call | Agent invokes external function |
| N/A | Tool result | Return value from tool |
| Quality filter | Verifier | Scoring logic |
| Keep/discard | Reward score | 0.0–1.0 continuous |
| Heuristic check | Verifier function | Programmatic scoring |
| Reward model | Judge model | LLM-based scoring |
| Dataset | Rollout dataset | Collection of rollouts |
| Training example | Demonstration | High-reward rollout for SFT |
| N/A | Preference pair | Two rollouts, one preferred (DPO) |
| Inference endpoint | Model server | LLM API |
| N/A | Resource server | Tools + verifiers |
| N/A | Agent server | Orchestration layer |
| Batch processing | Parallel rollout collection | Scale via parallelism |
| System prompt | Agent system prompt | Instructions for agent behavior |
| Few-shot examples | In-context examples | Examples in agent prompt |

