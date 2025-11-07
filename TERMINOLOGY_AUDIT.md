# Terminology Audit: Resource Server vs Environment vs Verifier

## Executive Summary

This audit identifies everywhere in the documentation where terminology is muddled between "resource server," "environment," and "verifier." The goal is to establish clear, consistent usage:

- **Resource Server** = Primary technical term (the HTTP server implementation)
- **Environment** = RL terminology mapping ONLY when explaining to RL practitioners
- **Verifier** = ONLY the `verify()` method/function, NOT the entire system

---

## Critical Files Requiring Updates

### 1. `/docs/resources/glossary.md`

**Current Issues:**
- Lines 50-51: Verifier definition is ambiguous ("may also refer colloquially")
- Line 12: Uses "agent-environment interactions" without clarifying what "environment" means
- Line 61: "Resource Server" defined separately but not mapped to RL terminology

**Recommended Changes:**

#### Change 1: Clarify Verifier Definition (lines 50-51)
```markdown
<!-- CURRENT -->
Verifier
  Component that scores agent outputs, producing reward signals. May also refer colloquially to "training environment with verifiable rewards."

<!-- RECOMMENDED -->
Verifier
Verification Logic
  The `verify()` method within a resource server that scores agent outputs and produces reward signals (typically 0.0-1.0). In reinforcement learning terminology, this is the **reward function**.

**Note**: Avoid using "verifier" to refer to the entire resource server—use "resource server" or "training environment" instead.
```

#### Change 2: Add RL Terminology Mapping Section
```markdown
<!-- ADD NEW SECTION AFTER LINE 68 -->

---

## RL Terminology Mapping

For practitioners familiar with reinforcement learning literature:

```{glossary}
Training Environment (RL)
  In NeMo Gym, **resource servers** serve as training environments. Each resource server implements:
  - **Action Space**: The tools/functions agents can call
  - **Reward Function**: The `verify()` endpoint that scores performance
  - **Task Distribution**: Datasets of problems to solve

Environment (Disambiguation)
  - **In RL context**: Use "training environment" or "resource server"
  - **In deployment context**: Development, staging, production environments (from configuration system)
  - **Avoid standalone "environment"** when referring to resource servers—be explicit
```
```

#### Change 3: Update Resource Server Definition (line 60-61)
```markdown
<!-- CURRENT -->
Resource Server
  Service that provides tools (functions agents can call) and verification logic (scoring agent performance).

<!-- RECOMMENDED -->
Resource Server
Training Environment
  HTTP service that provides tools (functions agents can call) and verification logic (scoring agent performance). In reinforcement learning terminology, resource servers implement **environments** with defined action spaces (tools) and reward functions (the `verify()` endpoint). Examples: `library_judge_math`, `comp_coding`, `google_search`.
```

---

### 2. `/docs/about/index.md`

**Current Issues:**
- Line 40: "Curated Environments" heading uses "environment" but means "resource servers"
- Line 41: "Each environment includes..." conflates terminology
- Line 54: "verifiers (logic to evaluate)" is accurate but could be clearer

**Recommended Changes:**

#### Change 1: Clarify "Curated Environments" section (lines 40-41)
```markdown
<!-- CURRENT -->
* - **Curated Environments**
  - A growing collection of resource servers across domains (mathematics, coding, knowledge, instruction-following, agent workflows) provides both the tools agents can use and the verification logic to score their performance. Each environment includes training datasets, validation sets, and examples.

<!-- RECOMMENDED -->
* - **Curated Training Environments**
  - A growing collection of resource servers across domains (mathematics, coding, knowledge, instruction-following, agent workflows) provides both the tools agents can use and the verification logic to score their performance. Each resource server includes training datasets, validation sets, and examples. In RL terminology, these are complete training environments with action spaces (tools) and reward functions (verification).
```

#### Change 2: Clarify Resources definition (line 54)
```markdown
<!-- CURRENT -->
* **Resources**: Servers that provide both tools (functions agents can call) and verifiers (logic to evaluate agent performance and assign reward signals). Examples include mathematical reasoning environments, code execution sandboxes, web search tools, and custom verification logic.

<!-- RECOMMENDED -->
* **Resources**: Servers that provide both tools (functions agents can call) and verification logic (methods that evaluate agent performance and assign reward signals). Examples include mathematical reasoning resource servers, code execution sandboxes, web search tools, and custom verification logic. In RL terminology, resource servers implement training environments.
```

#### Change 3: Add RL Terminology Note (after line 58)
```markdown
<!-- ADD AFTER LINE 58 -->

:::{note}
**For RL Practitioners**: In reinforcement learning terminology, NeMo Gym's "resource servers" are equivalent to "training environments." Each resource server defines an action space (available tools) and reward function (the `verify()` endpoint).
:::
```

---

### 3. `/docs/about/features.md`

**Current Issues:**
- Line 22: "Bridges the gap between environments and training infrastructure" - vague
- Line 23: "Production-Ready Environments" header
- Line 26: "Swap environments, models, and settings"
- Line 28: "make custom environments straightforward"
- Line 94: "Tutorial environment for getting started"
- Line 206: "Use same environments across different RL training implementations"
- Line 243: "Stateful Environments" header

**Recommended Changes:**

#### Change 1: Line 22
```markdown
<!-- CURRENT -->
* - **Framework Integration**
  - First-class support for popular RL frameworks. Bridges the gap between environments and training infrastructure.

<!-- RECOMMENDED -->
* - **Framework Integration**
  - First-class support for popular RL frameworks. Bridges the gap between resource servers (RL environments) and training infrastructure.
```

#### Change 2: Line 23
```markdown
<!-- CURRENT -->
* - **Production-Ready Environments**
  - Curated resource servers with verified accuracy. Not just toys—used for training NVIDIA Nemotron models.

<!-- RECOMMENDED -->
* - **Production-Ready Resource Servers**
  - Curated training environments with verified accuracy. Not just toys—used for training NVIDIA Nemotron models.
```

#### Change 3: Line 26
```markdown
<!-- CURRENT -->
* - **Configuration-First**
  - Swap environments, models, and settings via YAML. Minimal code changes for different training scenarios.

<!-- RECOMMENDED -->
* - **Configuration-First**
  - Swap resource servers, models, and settings via YAML. Minimal code changes for different training scenarios.
```

#### Change 4: Line 28
```markdown
<!-- CURRENT -->
* - **Extensible Foundation**
  - Clear abstractions and base classes make custom environments straightforward. Add new tools without understanding entire system.

<!-- RECOMMENDED -->
* - **Extensible Foundation**
  - Clear abstractions and base classes make custom resource servers straightforward. Add new tools without understanding entire system.
```

#### Change 5: Line 94
```markdown
<!-- CURRENT -->
* **Simple Weather**: Tutorial environment for getting started

<!-- RECOMMENDED -->
* **Simple Weather**: Tutorial resource server for getting started
```

#### Change 6: Line 206
```markdown
<!-- CURRENT -->
* - **OpenRLHF Compatible**
  - Trajectory format works with OpenRLHF framework. Use same environments across different RL training implementations.

<!-- RECOMMENDED -->
* - **OpenRLHF Compatible**
  - Trajectory format works with OpenRLHF framework. Use same resource servers (training environments) across different RL training implementations.
```

#### Change 7: Line 243
```markdown
<!-- CURRENT -->
* - **Stateful Environments**
  - Support for multi-turn conversations with session management. Maintain state across agent interactions for complex workflows.

<!-- RECOMMENDED -->
* - **Stateful Resource Servers**
  - Support for multi-turn conversations with session management. Maintain state across agent interactions for complex workflows.
```

---

### 4. `/docs/how-to-faq.md`

**Current Issues:**
- Line 38: "any business logic of tool implementations and verifiers" - uses "verifiers" to mean resource servers
- Line 367: "How To: Multi-verifier usage" - should be "Multi-resource-server"
- Line 368: "Gym is explicitly designed to support multi-verifier training"
- Line 370: "both math and search verifiers"
- Line 396: "For large scale verifier training"

**Recommended Changes:**

#### Change 1: Line 38
```markdown
<!-- CURRENT -->
Resource servers are used to abstract out any business logic of tool implementations and verifiers. Each resource server must implement a `verify` function.

<!-- RECOMMENDED -->
Resource servers are used to abstract out any business logic of tool implementations and verification logic. Each resource server must implement a `verify()` method that scores agent performance.
```

#### Change 2: Lines 367-370
```markdown
<!-- CURRENT -->
# How To: Multi-verifier usage
Gym is explicitly designed to support multi-verifier training.

Let's say you want to use both math and search verifiers. Normally how you spin up the servers individually is:

<!-- RECOMMENDED -->
# How To: Multi-Resource-Server Usage
Gym is explicitly designed to support using multiple resource servers (training environments) together.

Let's say you want to use both the math and search resource servers. Normally how you spin up the servers individually is:
```

#### Change 3: Line 396
```markdown
<!-- CURRENT -->
# How To: Profile your resources server
For large scale verifier training, it's critical that your resources server is as efficient as possible. It may be slammed with 16k concurrent requests or more. Gym provides easy tools to profile and understand the efficiency of your servers.

<!-- RECOMMENDED -->
# How To: Profile your resources server
For large scale training, it's critical that your resources server is as efficient as possible. It may be slammed with 16k concurrent requests or more. Gym provides easy tools to profile and understand the efficiency of your servers.
```

---

### 5. `/docs/about/concepts/core-abstractions.md`

**Current Issues:**
- Line 116: "Debug your math verifier" - should be "math resource server" or "math verification logic"

**Recommended Changes:**

#### Change 1: Line 116
```markdown
<!-- CURRENT -->
* - **Test in Isolation**
  - Debug your math verifier without touching the model. Test your agent logic without deploying infrastructure. Each piece works independently.

<!-- RECOMMENDED -->
* - **Test in Isolation**
  - Debug your math resource server's verification logic without touching the model. Test your agent logic without deploying infrastructure. Each piece works independently.
```

---

### 6. `/docs/about/concepts/rollout-collection-fundamentals.md`

**Current Issues:**
- Line 41: "The math verifier scoring it as correct"

**Recommended Changes:**

#### Change 1: Line 41
```markdown
<!-- CURRENT -->
- The math verifier scoring it as correct (reward: 1.0)

<!-- RECOMMENDED -->
- The math resource server's verification logic scoring it as correct (reward: 1.0)
```

---

### 7. `/docs/training/resource-servers/index.md`

**Current Issues:**
- Line 99: "Testing and prototyping environment" - technically correct but could be clearer

**Recommended Changes:**

#### Change 1: Line 99 (OPTIONAL - this is actually okay in context)
```markdown
<!-- CURRENT -->
  - workbench
  - Varies
  - Testing and prototyping environment

<!-- IF CHANGING (optional) -->
  - workbench
  - Varies
  - Testing and prototyping resource server
```

---

## Additional Recommendations

### 8. Create New Documentation Page: "RL Terminology Guide"

**Location**: `/docs/about/rl-terminology-mapping.md`

**Purpose**: Explicit mapping for RL practitioners coming from standard RL frameworks

**Outline**:
```markdown
# RL Terminology Mapping

If you're coming from reinforcement learning research or frameworks like Gymnasium, Stable Baselines, or RLlib, this guide maps standard RL terminology to NeMo Gym concepts.

## Core Mapping

| RL Term | NeMo Gym Term | Notes |
|---------|---------------|-------|
| **Environment** | **Resource Server** | The complete package of tools + verification |
| **Action Space** | **Tools** (from resource server) | Functions the agent can call |
| **Reward Function** | **`verify()` method** | Scoring logic that returns 0.0-1.0 |
| **Episode** | **Rollout** or **Trajectory** | Complete interaction sequence |
| **Reset** | **`seed_session()`** | Initialize environment state |
| **Step** | **Tool call** | Single action execution |

## Why Different Terms?

NeMo Gym uses "resource server" instead of "environment" for two reasons:

1. **Avoid confusion with deployment environments**: "Environment" already means dev/staging/prod in configuration contexts
2. **Clarity in HTTP architecture**: These are literally HTTP servers that provide resources

When reading NeMo Gym documentation:
- "Resource server" = What RL literature calls an "environment"
- "Verification logic" or "`verify()` method" = What RL literature calls "reward function"
- Avoid standalone "verifier" referring to the whole system

## Example Mapping

**Standard RL code pattern**:
```python
env = gym.make("MathProblems-v0")
obs = env.reset()
action = policy.select_action(obs)
obs, reward, done, info = env.step(action)
```

**NeMo Gym equivalent**:
```python
# Resource server provides tools and verify() method
response = agent.run(task)  # Agent calls tools as needed
verification = resource_server.verify(response)  # Returns reward
```

The key difference: NeMo Gym is designed for **data collection at scale**, not step-by-step RL training loops.
```

---

## Summary Statistics

**Files requiring updates**: 7
**Total changes needed**: ~20 specific edits
**Severity**:
- **Critical** (creates confusion): 8 changes
- **Important** (improves clarity): 10 changes  
- **Nice-to-have** (consistency): 2 changes

**Estimated effort**: 2-3 hours for a careful review and implementation

---

## Implementation Priority

### Phase 1: Critical Clarifications (Do First)
1. Update glossary with RL terminology mapping
2. Fix "multi-verifier" to "multi-resource-server" in FAQ
3. Add RL practitioner note to about/index.md

### Phase 2: Consistency Pass (Do Second)
4. Update all "environment" → "resource server" in features.md
5. Fix "math verifier" → "math resource server" in concepts
6. Update about/index.md "Curated Environments" section

### Phase 3: Documentation Enhancement (Do Third)
7. Create new RL Terminology Mapping guide
8. Add cross-references from key pages to terminology guide

---

## Testing the Changes

After making updates, verify:
1. No broken cross-references
2. Terminology is consistent within each page
3. RL practitioners can quickly understand mapping
4. New users don't encounter contradictory terms
5. Glossary serves as single source of truth

---

## Questions for Review

Before implementing, confirm:
1. Should we completely avoid "environment" except when explicitly mapping to RL terminology?
2. Is "training environment" acceptable as a synonym for "resource server"?
3. Do we want to keep "verifier" in conversational contexts (e.g., "the math verifier") or standardize to "math resource server"?
4. Should we add a terminology decision tree to help writers?



