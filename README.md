# AI Evaluation Workshop

Workshop materials for AI evaluation in finance: portfolio optimization, LLM translation evaluation, and Agent-to-Agent (A2A) evaluation with Langfuse observability.

## Workshop Overview

This workshop demonstrates three progressive approaches to AI evaluation in a financial context:

| Notebook | Topic | Evaluation Method |
|----------|-------|-------------------|
| **1. Quant Portfolio** | Mathematical optimization | Backtesting, Sharpe ratio, drawdown |
| **2. LLM Translation** | Narrative → config translation | Field accuracy, LLM-as-Judge, Langfuse experiments |
| **3. AI Agents (A2A)** | Agent-to-Agent evaluation | Green/Purple protocol, iterative scoring |

## Quick Start

```bash
cd claude-code-work

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -e .

# Configure API keys
cp .env.example .env  # Edit with your keys

# Run notebooks
jupyter notebook
```

## Environment Variables

Create a `.env` file:

```env
# Required
GEMINI_API_KEY=your-gemini-api-key

# Langfuse (for tracing and experiments)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Workshop Content

### Notebook 1: Quantitative Portfolio Optimization

Evaluate mathematical portfolio optimization methods:

- **Mean-Variance Optimization (MVO)**: min_volatility, max_sharpe, efficient_return
- **Hierarchical Risk Parity (HRP)**: Clustering-based allocation
- **Backtesting**: Historical performance validation
- **Metrics**: Sharpe ratio, max drawdown, volatility

### Notebook 2: LLM Translation Evaluation

Evaluate LLM ability to translate investor narratives to portfolio configurations:

- **Translation**: Natural language → structured JSON config
- **Field Accuracy**: Programmatic validation of extracted fields
- **LLM-as-Judge**: Using LLMs to evaluate translation quality
- **Langfuse Experiments**: Batch evaluation with native dataset/experiment features

```python
# Run experiment on Langfuse dataset
result = dataset.run_experiment(
    name="gemini-baseline",
    task=translation_task,
    evaluators=[field_accuracy_eval, llm_judge_eval]
)
```

### Notebook 3: Agent-to-Agent (A2A) Evaluation

Evaluate AI agents using the Green/Purple agent protocol:

- **Purple Agent**: Portfolio optimizer (agent being evaluated)
- **Green Agent**: Evaluator with RAG + web search tools
- **Iterative Communication**: Multi-round queries and responses
- **Scoring**: Dimension-based assessment (1-10 scale)

```python
# Run A2A evaluation with 5 rounds of communication
result = run_a2a_evaluation(
    task_description="Build a conservative portfolio",
    portfolio_agent=purple_agent,
    evaluator_agent=green_agent,
    max_rounds=5  # Green asks 5 follow-up questions
)
print(f"Score: {result.overall_score}/10")
print(f"Messages: {len(result.conversation)}")  # 11 messages
```

## Project Structure

```
claude-code-work/
├── 1_quant_portfolio_optimization.ipynb  # Portfolio optimization
├── 2_llm_translation_evaluation.ipynb    # LLM translation + Langfuse
├── 3_ai_agents_a2a_evaluation.ipynb      # A2A evaluation protocol
├── portfolio_optimizer.py                # MVO, HRP, backtesting
├── llm_utils.py                          # Translation, evaluation, experiments
├── agents.py                             # Green/Purple agents, A2A protocol
├── scenarios.json                        # Investor personas
├── evaluation_dataset.json               # Evaluation dataset + RAG knowledge
└── pyproject.toml                        # Dependencies
```

## Key Dependencies

- `langchain` + `langchain-google-genai` - Agent framework with Gemini
- `langfuse` - LLM observability, experiments, datasets
- `pypfopt` - Portfolio optimization
- `yfinance` - Market data

## PDF Exports

Pre-executed notebooks are available as PDFs in `claude-code-work/`:
- `1_quant_portfolio_optimization.pdf`
- `2_llm_translation_evaluation.pdf`
- `3_ai_agents_a2a_evaluation.pdf`

## License

MIT
