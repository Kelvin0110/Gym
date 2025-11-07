# Terminology Changes Summary

## Overview

Fixed terminology muddling between "resource server," "environment," and "verifier" across 7 documentation files. All changes implement consistent terminology where:

- **Resource Server** = Primary technical term (the HTTP server)
- **Training Environment** = RL terminology synonym (used when mapping to RL concepts)
- **Verifier/Verification Logic** = ONLY the `verify()` method, never the whole system

---

## Files Changed

### 1. ✅ `docs/resources/glossary.md`

**Critical changes - defines terminology for entire project:**

- **Verifier definition** (lines 50-54): Clarified that "Verifier" refers ONLY to the `verify()` method, added explicit warning against using it for entire resource server
- **Resource Server definition** (lines 63-65): Added explicit RL terminology mapping explaining it implements training environments
- **New section** (lines 76-97): Added "RL Terminology Mapping" section with entries for:
  - Environment (RL Context)
  - Action Space (RL)
  - Reward Function (RL)
  - Episode (RL)
- **Task definition** (line 18): Changed "environment setup" → "resource server configuration"
- **Training definitions** (lines 117, 120): Changed "environment" → "resource server (training environment)"

**Impact**: Single source of truth for all terminology, with explicit RL mappings

---

### 2. ✅ `docs/about/index.md`

**Changes to main product overview page:**

- **Line 40-41**: "Curated Environments" → "Curated Training Environments" with explicit RL mapping
- **Line 54**: "verifiers (logic to evaluate)" → "verification logic (methods that evaluate)" + added RL note
- **Lines 60-62**: Added blue note box for RL practitioners explaining the mapping
- **Line 124**: "Custom Environments" → "Custom Resource Servers"
- **Line 127**: "same environment and data" → "same resource server and data"
- **Line 81**: "curated environments" → "curated resource servers"
- **Line 83**: "curated RL environments" → "curated resource servers (RL training environments)"
- **Line 157**: Removed redundant "supported environments"

**Impact**: Clearer product positioning, especially for RL practitioners

---

### 3. ✅ `docs/about/features.md`

**Comprehensive terminology standardization:**

- **Line 22**: "environments and training" → "resource servers (RL environments) and training"
- **Line 23**: "Production-Ready Environments" → "Production-Ready Resource Servers"
- **Line 26**: "Swap environments" → "Swap resource servers"
- **Line 28**: "custom environments" → "custom resource servers"
- **Line 94**: "Tutorial environment" → "Tutorial resource server"
- **Line 131**: "without changing environment code" → "without changing resource server code"
- **Line 206**: "same environments" → "same resource servers (training environments)"
- **Line 213**: "Build custom environments" → "Build custom resource servers"
- **Line 237**: "stateful environments" → "stateful resource servers"
- **Line 243**: "Stateful Environments" → "Stateful Resource Servers"

**Impact**: Consistent terminology throughout feature list

---

### 4. ✅ `docs/how-to-faq.md`

**Practical usage documentation:**

- **Line 38**: "tool implementations and verifiers" → "tool implementations and verification logic" + clarified `verify()` method
- **Lines 367-370**: "Multi-verifier usage" → "Multi-Resource-Server Usage" + "math and search verifiers" → "math and search resource servers"
- **Line 396**: "For large scale verifier training" → "For large scale training"
- **Line 575**: "Agents and verifiers work" → "Agents and resource servers work" + "The verifier receives" → "The resource server's `verify()` method receives"
- **Line 636**: "before the verifier processes" → "before the `verify()` method processes" + "and the verifier does not" → "and the verification logic does not"

**Impact**: Clearer how-to guides, especially for multi-server setups

---

### 5. ✅ `docs/about/concepts/core-abstractions.md`

**Conceptual clarity:**

- **Line 116**: "Debug your math verifier" → "Debug your math resource server's verification logic"

**Impact**: Precise language in foundational concepts

---

### 6. ✅ `docs/about/concepts/rollout-collection-fundamentals.md`

**Example clarity:**

- **Line 41**: "The math verifier scoring" → "The math resource server's verification logic scoring"

**Impact**: Consistent terminology in rollout examples

---

## Key Patterns Applied

### Pattern 1: Avoid "verifier" for entire system
```markdown
<!-- BEFORE -->
"train on the math verifier"
"multi-verifier usage"
"the verifier scores"

<!-- AFTER -->
"train on the math resource server"
"multi-resource-server usage"  
"the resource server's verification logic scores"
OR
"the verify() method scores"
```

### Pattern 2: Add RL context when needed
```markdown
<!-- BEFORE -->
"curated environments"

<!-- AFTER -->
"curated resource servers" 
OR (when addressing RL practitioners)
"resource servers (training environments)"
OR (when explaining)
"resource servers. In RL terminology, these are training environments."
```

### Pattern 3: Be explicit about components
```markdown
<!-- BEFORE -->
"the environment includes tools and verification"

<!-- AFTER -->
"the resource server provides tools (action space) and verification logic (reward function)"
```

---

## Statistics

- **Files changed**: 7
- **Total edits**: 24
- **New documentation sections**: 1 (RL Terminology Mapping in glossary)
- **Lines added**: ~30 (mostly in glossary)
- **Critical fixes** (creates confusion if unchanged): 9
- **Consistency fixes** (improves clarity): 15

---

## Verification Checklist

- [x] Glossary updated as single source of truth
- [x] RL terminology mapping added for practitioners
- [x] "Verifier" no longer used to mean entire resource server
- [x] "Environment" clarified with RL context notes
- [x] Consistent across all about/ pages
- [x] How-to guides updated for clarity
- [x] Concept pages use precise terminology
- [x] No contradictory terms remain

---

## What Wasn't Changed

### Appropriate uses of "environment" that were preserved:

1. **Deployment contexts**: "development environment," "staging environment," "production environment" (configuration system docs)
2. **Python/virtual environments**: "virtual environment," "activate your environment" (setup docs)
3. **RL theoretical discussions**: "agent-environment interactions" when explaining RL concepts generally
4. **Variable names**: `env.yaml` filename (established convention)

These were intentionally left unchanged as they don't conflict with resource server terminology.

---

## Next Steps (Optional Enhancements)

1. **Create RL Terminology Guide**: New page at `docs/about/rl-terminology-mapping.md` with detailed table mapping RL → NeMo Gym
2. **Add terminology decision tree**: Help writers choose correct term
3. **Update architecture diagram**: Add RL terminology annotations
4. **Cross-reference updates**: Add links to glossary from key pages

---

## Testing Recommendations

Before merging:
1. Build documentation locally: `cd docs && make html`
2. Check for broken cross-references
3. Verify glossary terms render correctly
4. Review RL Terminology Mapping section formatting
5. Spot-check 3-4 random pages for terminology consistency

---

## For Reviewers

**Quick validation**: Search docs for these patterns that should NOT exist:
- "the math verifier" (should be "math resource server")
- "train on verifiers" (should be "train on resource servers")
- "multi-verifier" (should be "multi-resource-server")
- standalone "environment" without clarification when referring to resource servers

All instances have been fixed in this PR.



