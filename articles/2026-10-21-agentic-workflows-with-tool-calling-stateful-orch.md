---
tags: [agentic-ai, tool-calling, orchestration, state-machines, human-in-the-loop, langgraph, temporal, openai, anthropic]
---

![Agentic workflows with tool calling, stateful orchestration, and human-in-the-loop control](../images/agentic-workflows-with-tool-calling-sta.jpg)

# Designing Reliable Agentic Workflows: Tool Calling, Stateful Orchestration, and Human-in-the-Loop Control

**By Reallytics AI**

## TL;DR

- **Don’t build agent systems as “just prompts.”** In production, reliability comes from **explicit state machines**, typed tool schemas, idempotent side effects, and deterministic fallback paths.
- **Separate reasoning from execution.** Let the model decide *what* to do, but let a workflow engine decide *when*, *how often*, and *under what retries/guards* tools are invoked.
- **Human-in-the-loop is not a failure mode.** It’s a first-class state in the workflow, especially when confidence is low, actions are irreversible, or policy boundaries are involved.

## Introduction — Why This Matters Now

Teams are moving beyond single-turn chatbots and shipping systems that can **plan, call tools, retry, and escalate**. That shift is real, but it comes with an uncomfortable truth: the hard part is rarely the model prompt.

The production challenges are usually:

- **State drift** across multiple tool calls
- **Non-idempotent actions** causing duplicate writes or duplicate tickets
- **Unbounded retries** and runaway token/tool costs
- **Ambiguous failure modes** when the model hallucinates tool arguments
- **Unsafe autonomy** when an agent makes irreversible decisions too early

The winning pattern we keep seeing is:  
**LLM as planner + workflow engine as control plane + tools as bounded capabilities + human review as a state transition.**

That means agentic systems should be designed like distributed systems, not demos.

---

## 1) The Core Design Principle: Make the State Machine Explicit

A reliable agent pipeline should not “freestyle” through prompts. It should move through known states:

- `START`
- `NEED_CLARIFICATION`
- `PLAN_READY`
- `TOOL_PENDING`
- `TOOL_SUCCESS`
- `TOOL_FAILURE`
- `NEEDS_HUMAN_REVIEW`
- `DONE`
- `ABORTED`

This is where frameworks like **LangGraph**, **Temporal**, and **Prefect** fit naturally. The model can participate in transitions, but the transition logic should live in code.

### Example: a minimal explicit state machine

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional

class State(str, Enum):
    START = "start"
    PLAN_READY = "plan_ready"
    TOOL_PENDING = "tool_pending"
    TOOL_SUCCESS = "tool_success"
    TOOL_FAILURE = "tool_failure"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    DONE = "done"
    ABORTED = "aborted"

@dataclass
class AgentContext:
    user_request: str
    plan: Optional[list[str]] = None
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[dict[str, Any]] = None
    confidence: float = 0.0
    attempts: int = 0
    approved_by_human: bool = False

def transition(ctx: AgentContext, state: State) -> State:
    if state == State.START:
        # LLM creates a plan, but we require structured output
        if ctx.confidence < 0.6:
            return State.NEEDS_HUMAN_REVIEW
        return State.PLAN_READY

    if state == State.PLAN_READY:
        return State.TOOL_PENDING

    if state == State.TOOL_PENDING:
        return State.TOOL_SUCCESS if ctx.tool_result else State.TOOL_FAILURE

    if state == State.TOOL_FAILURE:
        return State.NEEDS_HUMAN_REVIEW if ctx.attempts >= 3 else State.TOOL_PENDING

    if state == State.NEEDS_HUMAN_REVIEW:
        return State.DONE if ctx.approved_by_human else State.ABORTED

    return state
```

The point isn’t the code itself — it’s the **discipline**: every path is explicit, inspectable, and observable.

---

## 2) Tool Schemas Must Be Strict, Typed, and Bounded

Tool calling is only reliable when the contract is tight. In production, “natural language arguments” are a liability. You want:

- **JSON Schema or Pydantic validation**
- **allowlisted tools only**
- **bounded parameters** (length, range, enum values)
- **clear side-effect semantics**
- **idempotency keys** for any write operation

This applies whether you’re using the **OpenAI Responses API**, **Anthropic tool use**, **LangChain**, or **LlamaIndex** tool wrappers.

### Example: a strongly typed tool schema

```python
from pydantic import BaseModel, Field, conint
from typing import Literal

class CreateRefundRequest(BaseModel):
    order_id: str = Field(min_length=8, max_length=64)
    amount_cents: conint(gt=0, le=500000)
    reason: Literal["duplicate_charge", "service_failure", "customer_request"]
    idempotency_key: str = Field(min_length=16, max_length=128)

def create_refund(req: CreateRefundRequest) -> dict:
    # Enforce idempotency at the service boundary
    existing = lookup_refund_by_idempotency_key(req.idempotency_key)
    if existing:
        return {"status": "already_processed", "refund_id": existing["refund_id"]}

    refund = payment_api.create_refund(
        order_id=req.order_id,
        amount_cents=req.amount_cents,
        reason=req.reason,
        idempotency_key=req.idempotency_key,
    )
    return {"status": "created", "refund_id": refund["id"]}
```

### Lessons learned

- If the model can invent a field, it will.
- If your tool can be called twice, it eventually will be.
- If your side effect isn’t idempotent, your incident will be expensive.

---

## 3) Human-in-the-Loop Should Be a First-Class Branch

Human review should not be bolted on as a manual exception queue. It should be a **workflow node** with clear entry criteria.

Common triggers:

- Low confidence in plan generation
- High-risk actions: refunds, deletions, access changes, compliance decisions
- Conflicting tool outputs
- Repeated tool failures
- Policy violations or prompt injection signals
- Missing required context

### Example: escalation gate

```python
def should_escalate(ctx: AgentContext) -> bool:
    risk_signals = [
        ctx.confidence < 0.7,
        ctx.attempts >= 2,
        ctx.tool_name in {"delete_user", "issue_refund", "grant_access"},
        ctx.tool_result is not None and ctx.tool_result.get("anomaly") is True,
    ]
    return any(risk_signals)

def run_agent_step(ctx: AgentContext) -> State:
    if should_escalate(ctx):
        notify_human_reviewer(ctx)
        return State.NEEDS_HUMAN_REVIEW

    # otherwise proceed with normal execution
    return State.TOOL_PENDING
```

In a real system, the human reviewer should see:

- the original request
- the current state
- the model’s rationale
- tool inputs/outputs
- audit trail
- a safe set of actions: **approve**, **edit**, **reject**, **request more context**

That’s how you keep humans in control without turning them into bottlenecks.

---

## 4) Reference Architecture: “Agent + Workflow Engine” in Production

Here’s the architecture we recommend for durable agentic systems:

```text
[User/UI]
   |
   v
[API Gateway] -----> [AuthN/AuthZ + Policy]
   |
   v
[Workflow Engine: Temporal / Prefect / LangGraph]
   |
   +--> [Planner Node: LLM Reasoning]
   |
   +--> [Tool Router]
   |        |
   |        +--> [Read-only Tools: search, retrieve, classify]
   |        +--> [Write Tools: ticketing, payments, CRM]
   |
   +--> [State Store: Postgres/Redis/S3]
   |
   +--> [Human Review Queue]
   |
   +--> [Observability: traces, prompts, tool calls, costs]
   |
   v
[Final Action / Response]
```

### Why this works

- **Workflow engine** owns retries, timeouts, persistence, compensation, and replay.
- **LLM** owns planning and interpretation, not orchestration.
- **State store** makes execution resumable and auditable.
- **Human review queue** handles exceptions and high-risk actions.
- **Observability** gives you cost, latency, and failure visibility per step.

### Practical implementation choices

- **Temporal**: best when you need durable execution, compensation, and long-running business workflows.
- **LangGraph**: excellent for graph-based agent state machines and LLM-driven transitions.
- **Prefect**: good for data/ML pipelines and operational workflows with simpler agent branching.
- **AutoGen**: useful for multi-agent collaboration patterns, but still needs production guardrails.
- **OpenAI Responses API / Anthropic tool use**: ideal for model-native tool invocation, but keep the control plane outside the model.

---

## 5) Production Lessons Learned

Here are the lessons that matter once you’re past the prototype:

### 1. Make every tool call replayable
If the workflow retries after a timeout, the tool must not create duplicate effects. Use:

- idempotency keys
- request hashes
- deduplication tables
- exactly-once semantics at the business boundary, not just in the workflow engine

### 2. Separate read and write tools
Let the model freely use read-only tools. Put stricter controls on write tools. In practice, we often require:

- human approval for destructive actions
- policy checks before execution
- a dry-run mode that previews impact

### 3. Log structured traces, not just text
Store:

- prompt version
- tool schema version
- model name and temperature
- inputs/outputs per tool
- latency per node
- escalation reason
- final decision

Without this, debugging agent failures is guesswork.

### 4. Assume prompt injection is real
Any external content can be adversarial: web pages, emails, support tickets, CRM notes. Treat retrieved text as **data**, not instruction.

### 5. Cap autonomy by risk tier
Not all actions deserve the same policy. A common pattern:

- **Tier 0**: summarize, classify, retrieve
- **Tier 1**: draft response, prepare ticket
- **Tier 2**: update internal records with approval
- **Tier 3**: destructive or financial actions require human approval

---

## Key Takeaways

- Design agent workflows as **state machines**, not loops of prompts.
- Treat tool calling as an **API contract**, with strict schemas and idempotency.
- Use a **workflow engine** for retries, persistence, and compensation.
- Make **human-in-the-loop** a normal state transition, not a fallback hack.
- Measure everything: **cost, latency, failure rates, tool reliability, and escalation frequency**.

---

## Further Reading

- LangGraph docs: https://langchain-ai.github.io/langgraph/
- LangChain tool calling: https://python.langchain.com/docs/concepts/tool_calling/
- OpenAI Responses API: https://platform.openai.com/docs/guides/responses
- Anthropic tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- LlamaIndex agents/tools: https://docs.llamaindex.ai/
- Temporal docs: https://docs.temporal.io/
- Prefect docs: https://docs.prefect.io/
- AutoGen GitHub: https://github.com/microsoft/autogen

If you’re building an agent in production, the model is only one component. The real product is the **orchestration layer around the model** — the state machine, policies, retries, and human controls that make the system trustworthy enough to ship.