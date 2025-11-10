(training-verification)=

# Verification

When you choose a resource server in NeMo Gym, you're also choosing its verification logic—the code that scores agent outputs and produces reward signals for training. This section helps you understand how that verification works, validate that it produces useful signals for your training algorithm, and customize it if needed.

---

## Verification Lives in Resource Servers

Verification is not a separate component you configure independently. When you select a resource server (like `mcqa`, `library_judge_math`, or `comp_coding`), that server already contains a `verify()` function that defines how to score agent outputs. You cannot verify without a resource server—they're the same architectural component.

**If you haven't chosen a resource server yet**, start with {ref}`training-resource-servers` to select one first. **If you already chose one**, this section shows you how to validate it works and customize it if needed.

---

## How the Architecture Works

Verification happens automatically during rollout collection through an HTTP endpoint. Each resource server implements a `verify()` method that becomes a `POST /verify` endpoint:

```python
# From nemo_gym/base_resources_server.py
class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/verify")(self.verify)  # HTTP endpoint
        return app
    
    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass  # Each resource server implements this
```

During rollout collection, the system automatically calls this endpoint after each agent interaction:

1. Agent completes interaction → produces output
2. Collection system calls `POST /verify` on your resource server
3. Resource server's `verify()` function scores the output
4. Reward signal gets saved in the rollout JSONL

### Simple Example: structured_outputs Server

The `structured_outputs` server validates that agent output matches a JSON schema:

```python
async def verify(self, body: VerifyRequest) -> BaseVerifyResponse:
    # Extract agent's response text
    response_text = extract_output_text(body.response)
    
    # Validate against JSON schema
    try:
        response_obj = json.loads(response_text)
        validate_against_schema(response_obj, body.schema_str)
        reward = 1.0  # Valid JSON
    except Exception:
        reward = 0.0  # Invalid JSON
    
    return BaseVerifyResponse(reward=reward)
```

### Complex Example: mcqa Server

The `mcqa` server extracts multiple-choice answers using configurable patterns:

```python
async def verify(self, body: MCQAVerifyRequest) -> MCQAVerifyResponse:
    # Extract agent's text response
    text = extract_last_assistant_text(body)
    
    # Parse answer using grading mode
    if body.grading_mode == "strict_single_letter_boxed":
        # Extract letter from \boxed{A} format
        predicted_letter = parse_boxed_answer(text, allowed_letters)
    elif body.grading_mode == "lenient_boxed":
        # Try multiple extraction patterns
        predicted_letter = parse_flexible(text, allowed_letters)
    
    # Compare with expected answer
    reward = 1.0 if predicted_letter == body.expected_answer else 0.0
    
    return MCQAVerifyResponse(reward=reward)
```

### What Each Server Verifies

Different resource servers implement fundamentally different verification approaches based on their task domains:

- **`mcqa`**: Extracts answer letters (A/B/C/D) using regex patterns and compares with expected answer
- **`comp_coding`**: Executes generated code against unit tests to check correctness
- **`library_judge_math`**: Validates mathematical equivalence using symbolic math libraries
- **`equivalence_llm_judge`**: Uses an LLM to judge semantic similarity between responses
- **`structured_outputs`**: Validates that output conforms to a JSON schema
- **`instruction_following`**: Checks that agent satisfied all specified constraints
- **`python_math_exec`**: Executes Python code and validates computational results

When you choose a resource server, you're choosing all of this verification logic together.

---

## What You Can Do Next

Now that you understand how verification works in NeMo Gym, you have two paths forward depending on your needs.

### Validate Your Resource Server's Verification

Most users start here. After selecting a resource server, you need to validate that its `verify()` function produces useful reward signals for your training algorithm. This involves collecting a small batch of test rollouts and checking that:

- The reward distribution is reasonable (not all 0.0 or 1.0)
- High-reward examples are actually good quality
- Low-reward examples are actually poor quality
- The scoring aligns with your intuition

See {doc}`validate-verification` for a step-by-step validation workflow (5-10 minutes).

### Build Custom Verification Logic

Advanced users who need domain-specific verification can implement their own `verify()` function in a custom resource server. This is useful when:

- None of the existing servers match your task domain
- You need multi-objective scoring (tracking multiple quality dimensions)
- Your verification logic requires specialized tools or libraries
- You want to combine multiple verification strategies

See {doc}`custom-patterns-cookbook` for copy-paste patterns and examples, or {doc}`multi-objective-scoring` for techniques that track multiple quality dimensions.

```{toctree}
:hidden:
:maxdepth: 1

validate-verification
custom-patterns-cookbook
multi-objective-scoring
```
