# AI Evaluation Workshop

Workshop materials for AI evaluation in finance: portfolio optimization, LLM translation, and AI agents with Langfuse tracing.

## Overview

This workshop demonstrates three key areas of AI evaluation:

1. **Portfolio Optimization** - Quantitative methods (MVO, HRP) with backtesting
2. **LLM Translation** - Converting investor narratives to portfolio configurations
3. **AI Agents** - Agent-to-Agent (A2A) evaluation with automatic tracing

## Notebooks

| Notebook | Description |
|----------|-------------|
| `1_quant_portfolio_optimization.ipynb` | Mean-Variance Optimization, HRP, efficient frontier |
| `2_llm_translation_evaluation.ipynb` | LLM translation from narrative to portfolio config with evaluation |
| `3_ai_agents_a2a_evaluation.ipynb` | Portfolio and evaluator agents with A2A protocol |

## Setup

### Prerequisites

- Python 3.10+
- API keys for Gemini (required), Anthropic, OpenAI (optional)

### Installation

```bash
cd claude-code-work

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GEMINI_API_KEY=your-gemini-api-key

# Optional (for multi-provider support)
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key

# Langfuse (for tracing)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Project Structure

```
claude-code-work/
├── 1_quant_portfolio_optimization.ipynb  # Portfolio optimization methods
├── 2_llm_translation_evaluation.ipynb    # LLM translation with evaluation
├── 3_ai_agents_a2a_evaluation.ipynb      # A2A evaluation protocol
├── portfolio_optimizer.py                # Portfolio optimization utilities
├── llm_utils.py                          # Translation utilities
├── agents.py                             # AI agents with Langfuse tracing
├── scenarios.json                        # Test scenarios and personas
├── evaluation_dataset.json               # Evaluation dataset
└── pyproject.toml                        # Project dependencies
```

## Key Features

### Automatic Langfuse Tracing

All agents use the `@observe()` decorator pattern for automatic hierarchical tracing:

```python
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

@observe()
def run_my_agent(agent, query, session_id=None):
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(name="my_agent", session_id=session_id)

    handler = CallbackHandler()  # Auto-inherits trace context
    result = agent.invoke({"input": query}, config={"callbacks": [handler]})
    return result
```

### Multi-Provider Support

Agents support multiple LLM providers:

```python
from agents import create_portfolio_agent, LLMProvider

# Use Gemini (default)
agent = create_portfolio_agent(provider=LLMProvider.GEMINI)

# Use Anthropic
agent = create_portfolio_agent(provider=LLMProvider.ANTHROPIC)

# Use OpenAI
agent = create_portfolio_agent(provider=LLMProvider.OPENAI)
```

### Agent-to-Agent (A2A) Evaluation

The A2A protocol evaluates agent outputs using another agent:

```python
from agents import run_a2a_evaluation

result = run_a2a_evaluation(
    task_description="Build a conservative portfolio",
    portfolio_agent=portfolio_agent,
    evaluator_agent=evaluator_agent,
    session_id="evaluation_session"
)
print(f"Overall Score: {result.overall_score}/10")
```

## Dependencies

Key dependencies (see `pyproject.toml` for full list):

- `langchain` - Agent framework
- `langchain-google-genai` - Gemini integration
- `langchain-anthropic` - Claude integration
- `langchain-openai` - OpenAI integration
- `langfuse` - LLM observability
- `pypfopt` - Portfolio optimization
- `yfinance` - Market data

## License

MIT
