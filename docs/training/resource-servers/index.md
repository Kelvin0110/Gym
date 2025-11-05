(training-resource-servers)=

# Resource Servers

Choose which resource server to use for your training task. Resource servers provide tools, datasets, and verification for specific domains.

:::{card}

**What you'll find here**:

Select from 13 built-in resource servers covering MCQA, math, code generation, open-ended QA, and more.

^^^

**Choose by**:
- Task type (what your agent needs to do)
- Training algorithm (SFT, DPO, PPO)
- Reward pattern (binary, continuous, multi-metric)

:::

---

## What Resource Servers Provide

Each resource server is a complete package:

**Tools**: Functions agents can call (weather API, search, code execution, etc.)

**Datasets**: Example tasks in JSONL format (training, validation, examples)

**Verification**: Scoring logic that evaluates agent performance and returns reward signals

**When you pick a resource server**, you're choosing all three components together.

---

## By Task Type

Choose based on what your agent needs to do:

```{list-table}
:header-rows: 1
:widths: 30 25 20 25

* - Task Type
  - Server
  - Reward Type
  - When to Use
* - **Multiple Choice (A/B/C/D)**
  - mcqa
  - Binary
  - Questions with single correct letter answer
* - **Math Problems**
  - library_judge_math
  - Continuous
  - Math with multiple valid forms (fractions, decimals, etc.)
* - **Code Generation**
  - comp_coding
  - Binary
  - Programming challenges with unit tests
* - **Open-Ended QA**
  - equivalence_llm_judge
  - Continuous
  - Questions with multiple valid phrasings
* - **Python Math Code**
  - python_math_exec
  - Binary
  - Math problems solved via code execution
* - **Instruction Following**
  - instruction_following
  - Binary
  - Tasks with specific constraints to satisfy
* - **JSON/Structured Output**
  - structured_outputs
  - Binary
  - Agent must produce valid JSON schema
* - **Multi-Needle Extraction**
  - multineedle
  - Binary + metrics
  - Extract multiple items from long context
* - **Agent Tool Use**
  - google_search
  - Varies
  - Search tasks with tool calling
* - **Agent Tool Use**
  - simple_weather
  - Varies
  - Weather queries with tool calling
* - **Stateful Interaction**
  - stateful_counter
  - Varies
  - Multi-turn state management
* - **Complex Agent Tasks**
  - multiverse_math_hard
  - Continuous
  - Multi-step reasoning with tools
* - **Custom Workbench**
  - workbench
  - Varies
  - Testing and prototyping environment
```

---

## By Training Algorithm

Choose based on what type of training you're doing:

### SFT (Supervised Fine-Tuning)

**What you need**: Binary or high-threshold rewards to filter for correct examples only

**Recommended servers**:
- `mcqa` - Multiple choice with exact matching
- `comp_coding` - Code that passes all tests
- `instruction_following` - All constraints satisfied
- `python_math_exec` - Correct computational result

**Why these work**: Clean correct/incorrect distinction makes it easy to filter for high-quality training examples. You'll filter for `reward >= 0.95` or `reward == 1.0`.

**Example workflow**:
```bash
# Collect rollouts with binary verification
ng_collect_rollouts +resource_server=mcqa ...

# Filter for correct examples only
# (See prepare-for-training.md for code)
```

---

### DPO (Direct Preference Optimization)

**What you need**: Continuous rewards with clear quality separation between responses

**Recommended servers**:
- `library_judge_math` - Partial credit for math quality
- `equivalence_llm_judge` - Semantic similarity scoring
- `multineedle` - Multi-part extraction with overlap scoring

**Why these work**: Continuous rewards (0.0–1.0 range) let you create preference pairs with clear winner/loser. You need minimum 0.2 gap between chosen and rejected.

**Example workflow**:
```bash
# Collect rollouts with continuous verification
ng_collect_rollouts +resource_server=library_judge_math ...

# Create pairs with quality gap >= 0.2
# (See prepare-for-training.md for code)
```

---

### PPO/RL

**What you need**: Rich continuous signal with varied distribution

**Recommended servers**:
- `library_judge_math` - Hybrid symbolic + judge verification
- `multineedle` - Multiple metrics for analysis
- `equivalence_llm_judge` - Semantic similarity gradients
- `multiverse_math_hard` - Complex multi-step reasoning

**Why these work**: Continuous rewards provide nuanced quality signals for policy gradient learning. Distribution should spread across 0.0–1.0 with std dev > 0.15.

**Example workflow**:
```bash
# Collect rollouts with rich verification
ng_collect_rollouts +resource_server=library_judge_math ...

# Validate distribution is varied
# (See validate-verification.md for checks)
```

---

## Understanding Reward Types

### Binary (0.0 or 1.0)

**Characteristics**:
- Only two values: correct (1.0) or incorrect (0.0)
- Deterministic and unambiguous
- Fast verification (simple checks)

**Servers**: `mcqa`, `comp_coding`, `instruction_following`, `python_math_exec`

**Best for**: SFT data filtering

**Training implications**: 
- ✅ Clean training data (only correct examples)
- ❌ Limited signal for DPO/PPO (no quality gradations)

**Example reward distribution**:
```
Rewards: [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
Mean: 0.57 (57% correct)
```

---

### Continuous (0.0–1.0 range)

**Characteristics**:
- Scores across full range (partial credit)
- Nuanced quality assessment
- Captures degrees of correctness

**Servers**: `library_judge_math`, `equivalence_llm_judge`, `multiverse_math_hard`

**Best for**: DPO preference pairs, PPO gradient learning

**Training implications**:
- ✅ Rich signal for quality differences
- ✅ Can create preference pairs
- ⚠️ May need calibration for consistent scoring

**Example reward distribution**:
```
Rewards: [0.85, 0.42, 0.91, 0.33, 0.67, 0.18, 0.88]
Mean: 0.61
Std Dev: 0.28 (good variance)
```

---

### Multi-Metric

**Characteristics**:
- Primary reward + additional tracking metrics
- Multiple quality dimensions
- Richer analysis capabilities

**Servers**: `multineedle` (accuracy + set_overlap), `library_judge_math` (library + judge)

**Best for**: Multi-objective optimization, detailed analysis

**Training implications**:
- ✅ Track trade-offs between objectives
- ✅ Filter by multiple criteria
- ⚠️ Primary reward still used for training

**Example multi-metric output**:
```json
{
  "reward": 0.82,          // Primary training signal
  "accuracy": 0.82,        // Binary correctness
  "set_overlap": 0.95      // Partial credit
}
```

---

## Server Comparison

Quick comparison of most commonly used servers:

```{list-table}
:header-rows: 1
:widths: 20 15 15 25 25

* - Server
  - Reward Type
  - Speed
  - Best Training Use
  - Example Tasks
* - **mcqa**
  - Binary
  - Very Fast
  - SFT
  - MMLU, ARC, HellaSwag
* - **library_judge_math**
  - Continuous
  - Fast → Slow
  - DPO, PPO
  - MATH, GSM8K, Minerva
* - **comp_coding**
  - Binary
  - Medium
  - SFT
  - HumanEval, MBPP
* - **equivalence_llm_judge**
  - Continuous
  - Slow
  - DPO, PPO
  - NaturalQuestions, TruthfulQA
* - **instruction_following**
  - Binary
  - Fast
  - SFT
  - IFEval, constraint tasks
* - **multineedle**
  - Binary + metrics
  - Fast
  - PPO, Analysis
  - Multi-hop QA, extraction
```

**Speed considerations**:
- **Very Fast** (< 10ms): Simple string matching, exact checks
- **Fast** (10-50ms): Symbolic verification, regex extraction
- **Medium** (50-500ms): Code execution, complex parsing
- **Slow** (> 500ms): LLM judge calls, expensive operations

For production scale (millions of rollouts), prefer fast verification when possible.

---

## Server Details and Configuration

Each resource server includes:

**📁 In `resources_servers/<server_name>/`**:
- `README.md` - Task description and usage
- `app.py` - Verification implementation
- `configs/*.yaml` - Configuration options
- `data/*.jsonl` - Example datasets
- `test_*.py` - Test suite

**Common configuration options**:

**mcqa**:
- `grading_mode`: `strict_single_letter_boxed`, `lenient_boxed`, `lenient_answer_colon`
- `template_metadata.output_regex`: Custom extraction patterns

**library_judge_math**:
- `should_use_judge`: Enable/disable LLM judge fallback
- `judge_model_server`: Which model to use for judging

**equivalence_llm_judge**:
- `judge_system_message`: Custom judge instructions
- `judge_prompt_template`: Custom prompt format
- `check_twice_swap`: Bias reduction via position swapping

**comp_coding**:
- `num_processes`: Parallelism for test execution
- `unit_test_timeout_secs`: Maximum time per test

See each server's README for complete configuration details.

---

## Server READMEs

Browse individual server documentation:

- [mcqa](../../../resources_servers/mcqa/README.md) - Multiple choice questions
- [library_judge_math](../../../resources_servers/library_judge_math/README.md) - Math with symbolic + judge
- [comp_coding](../../../resources_servers/comp_coding/README.md) - Code generation with tests
- [equivalence_llm_judge](../../../resources_servers/equivalence_llm_judge/README.md) - Semantic QA judging
- [python_math_exec](../../../resources_servers/python_math_exec/README.md) - Math via code execution
- [instruction_following](../../../resources_servers/instruction_following/README.md) - Constraint checking
- [structured_outputs](../../../resources_servers/structured_outputs/README.md) - JSON schema validation
- [multineedle](../../../resources_servers/multineedle/README.md) - Multi-item extraction

Or browse all servers: [resources_servers/](../../../resources_servers/)


---

## Next Steps

**After selecting a resource server**:

:::{button-ref} ../rollout-collection/index
:color: primary
:outline:
:ref-type: doc

Start Collecting Rollouts →
:::

**After collecting samples**:

:::{button-ref} ../verification/validate-verification
:color: secondary
:outline:
:ref-type: doc

Validate Verification →
:::

**For custom domains**: See {doc}`../verification/custom-patterns-cookbook` to build custom verification.

