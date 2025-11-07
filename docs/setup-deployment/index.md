(setup-deployment)=

# Setup & Deployment

Configure and deploy NeMo Gym for your project. 

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New? Try Get Started
:::

---

## Configuration Management

Master NeMo Gym's three-tier configuration system to handle different environments, secrets, and deployment scenarios.

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Management
:link: configuration/index
:link-type: doc

Manage NeMo Gym's three-tier configuration system, handle environments, secrets, and multi-server setups.
+++
{bdg-secondary}`how-to` {bdg-secondary}`configuration`
:::

::::

:::{seealso}
**Understanding the concepts?** See {doc}`../about/concepts/configuration-system` for conceptual explanation of how the three-tier configuration system works.
:::

---

## Deployment

Deploy NeMo Gym in different environments—local development, remote servers, or containerized infrastructure.

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Deployment
:link: deployment/index
:link-type: doc

Deploy NeMo Gym locally, on remote servers, in containers, and scale for production.
+++
{bdg-secondary}`how-to` {bdg-secondary}`deployment`
:::

::::

---

## Operations

Monitor, test, and debug your NeMo Gym deployment to ensure reliable operation.

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Operations
:link: operations/index
:link-type: doc

Monitor, test, and debug your NeMo Gym deployment for reliable operation.
+++
{bdg-secondary}`how-to` {bdg-secondary}`operations`
:::

::::

---

## Common Workflows

End-to-end workflows for typical setup and deployment scenarios:

### Development Environment Setup

```bash
# 1. Clone and install
git clone <repo>
cd Gym
pip install -e ".[dev]"

# 2. Create env.yaml with secrets
cat > env.yaml << EOF
policy_api_key: sk-your-openai-key
EOF

# 3. Test with simple config
ng_run "+config_paths=[responses_api_agents/simple_agent/config.yaml]"
```

**Next**: See {doc}`configuration/index` for more configuration patterns

---

### Production Deployment

```bash
# 1. Set up production env.yaml
cat > env.yaml << EOF
policy_api_key: ${PROD_OPENAI_KEY}
judge_api_key: ${PROD_JUDGE_KEY}
EOF

# 2. Deploy with production config
ng_run "+config_paths=[production_config.yaml]" \
    +default_host=0.0.0.0 \
    +head_server.port=8000
```

**Next**: See {doc}`deployment/index` for deployment options

