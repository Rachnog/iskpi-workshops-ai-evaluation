# AI Evaluation in Finance Workshop

Hands-on workshop exploring AI evaluation techniques through portfolio optimization. Three progressive notebooks cover mathematical models, LLM translation, and Agent-to-Agent (A2A) evaluation with Langfuse observability.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Configure API keys in .env file
cp .env.example .env  # Edit with your keys

# Run notebooks
jupyter notebook
```

## Investor Personas

All notebooks use consistent investor personas for evaluation:

| Persona | Profile | Strategy |
|---------|---------|----------|
| **Sarah Chen** | 58yo, $500K, 7yr horizon, low risk | min_volatility, conservative universe |
| **Marcus Johnson** | 28yo, $50K, 30yr horizon, high risk | max_sharpe, us_tech universe |
| **Elena Rodriguez** | 38yo, $150K, 18yr horizon, medium risk | max_sharpe, global_diversified, 15% max position |

## Project Structure

```
├── 1_quant_portfolio_optimization.ipynb  # MVO, HRP, backtesting
├── 2_llm_translation_evaluation.ipynb    # LLM translation + Langfuse experiments
├── 3_ai_agents_a2a_evaluation.ipynb      # A2A protocol with Green/Purple agents
├── 4_skills_based_agents.ipynb           # Skills architecture comparison
├── portfolio_optimizer.py                # Portfolio optimization functions
├── llm_utils.py                          # Translation and evaluation utilities
├── agents.py                             # Agent definitions, Skills, A2A protocol
├── .claude/skills/                       # Skills definitions
│   ├── universe-selection/SKILL.md
│   ├── optimization-execution/SKILL.md
│   ├── risk-assessment/SKILL.md
│   ├── backtesting/SKILL.md
│   └── portfolio-comparison/SKILL.md
├── scenarios.json                        # Investor personas and test scenarios
├── evaluation_dataset.json               # Evaluation data + RAG knowledge base
└── pyproject.toml                        # Package configuration
```

## Environment Setup

Create `.env` file:

```env
# Required
GEMINI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key  # For Skills-based agent (Notebook 4)

# Langfuse (for tracing and experiments)
LANGFUSE_SECRET_KEY=your-key
LANGFUSE_PUBLIC_KEY=your-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Notebook 1: Quantitative Portfolio Optimization

Evaluate mathematical portfolio optimization methods using backtesting and financial metrics.

**Topics:** Mean-Variance Optimization, Hierarchical Risk Parity, Efficient Frontier, Backtesting

**Key Functions:**
```python
from portfolio_optimizer import optimize_portfolio, optimize_hrp, backtest_portfolio

# Mean-Variance Optimization
result = optimize_portfolio(config, prices)  # min_volatility, max_sharpe, efficient_return

# Hierarchical Risk Parity
result = optimize_hrp(prices)

# Backtest
metrics = backtest_portfolio(weights, prices)  # sharpe, drawdown, returns
```

**Evaluation Metrics:**
| Metric | Good Value | Description |
|--------|------------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Max Drawdown | > -20% | Worst peak-to-trough |
| Volatility | < 15% | Annualized std dev |

---

## Notebook 2: LLM Translation Evaluation

Evaluate LLM ability to translate investor narratives into structured portfolio configurations.

**Topics:** LLM Translation, Field Accuracy, LLM-as-Judge, Langfuse Datasets & Experiments

**Key Functions:**
```python
from llm_utils import translate_narrative, llm_as_judge, LLMProvider

# Translate narrative to config
result = translate_narrative(narrative, provider=LLMProvider.GEMINI)

# LLM-as-Judge evaluation
scores = llm_as_judge(narrative, output, expected, provider=LLMProvider.GEMINI)
```

**Langfuse Experiments:**
```python
from langfuse import Langfuse
from langfuse.evaluation import Evaluation

# Create dataset in Langfuse
langfuse = Langfuse()
dataset = langfuse.create_dataset(name="portfolio-translation-v1")

# Define task and evaluators
def translation_task(*, item, **kwargs):
    return translate_narrative(item.input["narrative"])["config"]

def field_accuracy_eval(*, output, expected_output, **kwargs):
    accuracy = compute_accuracy(output, expected_output)
    return Evaluation(name="field_accuracy", value=accuracy)

# Run experiment
result = dataset.run_experiment(
    name="gemini-baseline",
    task=translation_task,
    evaluators=[field_accuracy_eval, llm_judge_eval]
)
```

---

## Notebook 3: AI Agents & A2A Evaluation

Evaluate AI agents using the Agent-to-Agent (A2A) protocol with iterative communication.

**Topics:** LangChain Agents, RAG, Green/Purple Agent Protocol, Iterative Evaluation

**A2A Protocol Overview:**

The A2A protocol uses two agent roles:
- **Purple Agent** (Portfolio): The agent being evaluated
- **Green Agent** (Evaluator): Queries Purple agent and produces scores

```
A2A Protocol Flow (max_rounds=3)
────────────────────────────────
Round 1: GREEN → PURPLE (initial request)
         PURPLE → GREEN (portfolio recommendation)

Round 2: GREEN → PURPLE (follow-up question)
         PURPLE → GREEN (clarification)

Round 3: GREEN → PURPLE (probe deeper)
         PURPLE → GREEN (justification)

Final:   GREEN produces assessment (scores + feedback)
```

**Key Functions:**
```python
from agents import (
    create_a2a_purple_agent,
    create_a2a_green_agent,
    run_a2a_evaluation,
    create_rag_knowledge_base,
    LLMProvider
)

# Create agents
purple_agent = create_a2a_purple_agent(provider=LLMProvider.GEMINI)
retriever = create_rag_knowledge_base(eval_data)
green_agent = create_a2a_green_agent(retriever, provider=LLMProvider.GEMINI)

# Run A2A evaluation with iterative communication
result = run_a2a_evaluation(
    task_description="Build a conservative portfolio for retirement",
    portfolio_agent=purple_agent,
    evaluator_agent=green_agent,
    max_rounds=5,  # 5 rounds of communication
    session_id="evaluation_001"
)

# Results
print(f"Overall Score: {result.overall_score}/10")
print(f"Messages exchanged: {len(result.conversation)}")
for dim, score in result.scores.items():
    print(f"  {dim}: {score}/10")
```

**Green Agent Tools:**
- `search_knowledge_base` - RAG retrieval from evaluation dataset
- `web_search` - DuckDuckGo search for current market information

**Scoring Dimensions:**
| Dimension | Description |
|-----------|-------------|
| Universe Selection | Appropriate asset universe for investor |
| Optimization Method | Suitable optimization approach |
| Risk Assessment | Proper risk evaluation |
| Constraint Handling | Respect for investor constraints |
| Explanation Quality | Clear reasoning and trade-offs |

---

## Notebook 4: Skills-Based Agents

Demonstrates Claude's Agent Skills architecture using Anthropic SDK with native bash tool.

**Topics:** Native Skills Architecture, Bash Tool Execution, Tools vs Skills Comparison, A2A Evaluation

**Skills Architecture:**

Skills are modular capabilities defined in `.claude/skills/` directory:

```
.claude/skills/
├── universe-selection/SKILL.md    # Asset universe selection with decision matrix
├── optimization-execution/SKILL.md # MVO/HRP with ready-to-run scripts
├── risk-assessment/SKILL.md       # Risk metrics evaluation
├── backtesting/SKILL.md           # Historical validation
└── portfolio-comparison/SKILL.md  # Compare alternatives
```

**SKILL.md Format:**
```markdown
---
name: optimization-execution
description: Execute portfolio optimization using MVO or HRP...
---

# Portfolio Optimization Skill

## Quick Decision Guide
| Investor Profile | Method | Target |
|-----------------|--------|--------|
| Conservative | MVO | min_volatility |
| Balanced | MVO | max_sharpe |

## Complete Ready-to-Run Script
[Python code that can be copied and executed...]
```

**Key Functions:**
```python
from agents import (
    SkillLoader,
    create_native_skills_agent,
    run_native_skills_agent,
    run_native_skills_a2a_evaluation
)

# Load skills metadata
skill_loader = SkillLoader()
print(skill_loader.get_skill_descriptions())

# Create Native Skills Agent (uses Anthropic Claude with bash tool)
native_agent = create_native_skills_agent()

# Agent workflow:
# 1. Reads skill: cat .claude/skills/optimization-execution/SKILL.md
# 2. Runs code: python3 -c "from portfolio_optimizer import ..."
# 3. Explains results to investor

result = run_native_skills_agent(native_agent, "Build a conservative portfolio")

# A2A evaluation comparing Skills vs Tools
a2a_result = run_native_skills_a2a_evaluation(
    task_description="Build a portfolio",
    native_agent_config=native_agent,
    evaluator_agent=green_agent,
    max_rounds=3
)
```

**Tools vs Skills Comparison:**

| Aspect | Tools-based | Native Skills |
|--------|-------------|---------------|
| LLM | Gemini | Anthropic Claude |
| Tool calls | Function calls | bash (read files + run code) |
| Instructions | In code/docstrings | SKILL.md files |
| Code execution | Wrapper functions | Direct Python via bash |
| Updates | Code changes | Edit markdown files |

---

## Evaluation Summary

| Notebook | Evaluation Type | Key Metrics |
|----------|-----------------|-------------|
| 1. Quant | Backtesting | Sharpe > 1.0, Drawdown > -20% |
| 2. LLM | Field Accuracy + LLM Judge | Accuracy > 90%, Judge > 8/10 |
| 3. Agents | A2A Protocol | Overall Score > 7/10 |
| 4. Skills | Skills A2A Protocol | Skills Usage Score, Comparison |

## Langfuse Tracing

All LLM calls and agent interactions are traced in Langfuse using the `@observe()` decorator:

```python
from langfuse import observe

@observe()
def my_function():
    # All LLM calls inside are automatically traced
    result = llm.invoke(prompt)
    return result
```

## Troubleshooting

- **API errors**: Verify `.env` keys are set correctly
- **Gemini rate limits**: Add delays between calls or use batch processing
- **Agent timeout**: Increase timeout or reduce max_rounds
- **Memory issues**: Use `ledoit_wolf` covariance estimation

## References

- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [Langfuse Experiments](https://langfuse.com/docs/evaluation/experiments)
- [A2A Protocol](https://google.github.io/A2A/)
