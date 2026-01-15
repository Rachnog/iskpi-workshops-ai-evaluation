# AI Evaluation in Finance Workshop

This workshop explores AI evaluation techniques in finance through portfolio optimization. Learn to evaluate mathematical models, LLM translations, and AI agents using three progressive notebooks with consistent investor personas.

## Quick Start

```bash
# Install with uv
uv venv && source .venv/bin/activate
uv pip install -e .

# Or with pip
pip install -e .

# Configure API keys in .env file, then run:
jupyter notebook
```

## Three Consistent Personas

All notebooks use the same investor personas for consistent evaluation:

| Persona | Age | Investment | Horizon | Risk | Strategy |
|---------|-----|------------|---------|------|----------|
| **Sarah Chen** | 58 | $500,000 | 7 years | Low | min_volatility, conservative |
| **Marcus Johnson** | 28 | $50,000 | 30 years | High | max_sharpe, us_tech |
| **Elena Rodriguez** | 38 | $150,000 | 18 years | Medium | max_sharpe, global, 15% max |

## Project Structure

```
├── pyproject.toml           # uv package configuration
├── scenarios.json           # Investor personas and scenarios
├── evaluation_dataset.json  # LLM evaluation dataset (20 examples)
├── portfolio_optimizer.py   # Core portfolio optimization
├── llm_utils.py            # LLM translation utilities
├── agents.py               # Agent definitions
├── 1_quant_portfolio_optimization.ipynb
├── 2_llm_translation_evaluation.ipynb
└── 3_ai_agents_a2a_evaluation.ipynb
```

## Environment Setup

Create `.env` file:
```env
ANTHROPIC_API_KEY=your-key
GEMINI_API_KEY=your-key
LANGFUSE_SECRET_KEY=your-key
LANGFUSE_PUBLIC_KEY=your-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

---

## Notebook 1: Quantitative Portfolio Optimization

**Topics:** Mean-Variance Optimization, HRP, Backtesting, Financial Metrics

**Key Functions:**
- `optimize_portfolio()` - MVO with min_volatility, max_sharpe, efficient_return
- `optimize_hrp()` - Hierarchical Risk Parity
- `backtest_portfolio()` - Historical performance

### Exercise 1.1: Constraint Analysis
Compare Elena's portfolio with different max_position values (10%, 15%, 20%):
```python
for max_pos in [0.10, 0.15, 0.20]:
    config = PortfolioConfig(..., max_position=max_pos)
    result = optimize_portfolio(config, prices)
    print(f"Max {max_pos:.0%}: Sharpe={result['sharpe_ratio']:.3f}")
```

### Exercise 1.2: Covariance Methods
Compare 'sample', 'ledoit_wolf', and 'exp_cov' methods. Which is most stable?

---

## Notebook 2: LLM Translation Evaluation

**Topics:** LLM Translation, Multi-provider Comparison, LLM-as-Judge, Langfuse

**Key Functions:**
- `translate_narrative()` - Narrative → portfolio config
- `compute_field_accuracy()` - Measure accuracy
- `llm_as_judge()` - LLM evaluation

### Exercise 2.1: Provider Comparison
Run translations with Anthropic, OpenAI, and Gemini. Compare accuracy:
```python
for provider in [LLMProvider.ANTHROPIC, LLMProvider.GEMINI]:
    result = translate_narrative(narrative, provider)
    accuracy = compute_field_accuracy(result, expected)
    print(f"{provider.value}: {accuracy['overall_accuracy']:.1%}")
```

### Exercise 2.2: Custom Judge
Create an LLM judge that evaluates financial appropriateness and risk alignment.

---

## Notebook 3: AI Agents & A2A Evaluation

**Topics:** LangChain Agents, RAG, Agent-to-Agent Protocol, Skills Architecture

**Key Functions:**
- `create_portfolio_agent()` - Agent with portfolio tools
- `create_evaluator_agent()` - RAG-enhanced evaluator
- `run_a2a_evaluation()` - A2A protocol
- `run_skills_agent()` - Gemini function-calling agent

### Exercise 3.1: RAG Enhancement
Add examples to `evaluation_dataset.json` and measure evaluation improvement.

### Exercise 3.2: Architecture Comparison
Run the same scenario with standard agent vs skills agent. Compare:
- Task completion rate
- Number of tool calls
- Response quality

---

## Evaluation Metrics

| Category | Metric | Good Value |
|----------|--------|------------|
| Financial | Sharpe Ratio | > 1.0 |
| Financial | Max Drawdown | > -20% |
| Translation | Field Accuracy | > 90% |
| Translation | LLM Judge Score | > 8/10 |
| Agent | A2A Score | > 7/10 |

## Key Concepts

### Optimization Targets
- `min_volatility` - Minimize risk (Sarah)
- `max_sharpe` - Risk-adjusted returns (Elena, Marcus)
- `efficient_return` - Target return with min risk

### Asset Universes
- `conservative` - Bonds, low-risk (Sarah)
- `us_tech` - Technology stocks (Marcus)
- `global_diversified` - Global ETFs (Elena)

## Troubleshooting

- **API errors**: Check `.env` keys; Gemini is fallback
- **Memory issues**: Use `ledoit_wolf` covariance
- **Agent timeout**: Increase timeout, use verbose=True

## References

- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [Langfuse](https://langfuse.com/docs)
