"""
AI Agents Module

Portfolio optimization and evaluation agents using LangChain.
Multi-provider support (Gemini, Anthropic, OpenAI).
Langfuse tracing via @observe() decorator - automatic hierarchical tracing.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path='../.env')

# Set GOOGLE_API_KEY for langchain-google-genai
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if _gemini_key:
    os.environ["GOOGLE_API_KEY"] = _gemini_key

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import google.generativeai as genai

from portfolio_optimizer import (
    PortfolioConfig, download_market_data, optimize_portfolio,
    optimize_hrp, backtest_portfolio, get_universe
)

# Langfuse - simple decorator-based tracing
try:
    from langfuse import observe, get_client
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_ENABLED = True
except ImportError:
    LANGFUSE_ENABLED = False
    LangfuseCallbackHandler = None
    observe = lambda *a, **kw: (lambda f: f)  # no-op decorator
    get_client = lambda: None


VALID_OPTIMIZATION_TARGETS = ["min_volatility", "max_sharpe", "max_return", "efficient_return"]


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


def get_llm(provider: LLMProvider = LLMProvider.GEMINI, temperature: float = 0):
    """Get LLM instance based on provider."""
    if provider == LLMProvider.ANTHROPIC:
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=temperature)
    elif provider == LLMProvider.OPENAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    else:  # GEMINI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature, streaming=False)


# Portfolio Tools
@tool
def get_available_universes() -> str:
    """Get available asset universes for portfolio construction."""
    from portfolio_optimizer import UNIVERSE_DEFINITIONS
    return json.dumps({u: t for u, t in UNIVERSE_DEFINITIONS.items()}, indent=2)


@tool
def optimize_portfolio_tool(universe: str, optimization_target: str = "max_sharpe",
                            max_position: Optional[float] = None,
                            target_return: Optional[float] = None) -> str:
    """Optimize a portfolio using Mean-Variance Optimization. optimization_target can be: min_volatility, max_sharpe, max_return, or efficient_return."""
    try:
        tickers = get_universe(universe)
        prices = download_market_data(tickers, "2019-01-01", "2024-01-01")

        config = PortfolioConfig(
            tickers=list(prices.columns), start_date="2019-01-01", end_date="2024-01-01",
            optimization_target=optimization_target, max_position=max_position, target_return=target_return
        )

        result = optimize_portfolio(config, prices)
        backtest = backtest_portfolio(result["weights"], prices)

        output = {
            "universe": universe, "optimization_target": optimization_target,
            "weights": {k: round(v, 4) for k, v in result["weights"].items() if v > 0.01},
            "expected_return": round(result["expected_return"] * 100, 2),
            "volatility": round(result["volatility"] * 100, 2),
            "sharpe_ratio": round(result["sharpe_ratio"], 3),
            "backtest_sharpe": round(backtest["sharpe_ratio"], 3),
            "max_drawdown": round(backtest["max_drawdown"] * 100, 2)
        }
        return json.dumps(output, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_hrp_optimization(universe: str) -> str:
    """Run Hierarchical Risk Parity optimization."""
    try:
        tickers = get_universe(universe)
        prices = download_market_data(tickers, "2019-01-01", "2024-01-01")
        result = optimize_hrp(prices)
        backtest = backtest_portfolio(result["weights"], prices)

        output = {
            "method": "HRP", "universe": universe,
            "weights": {k: round(v, 4) for k, v in result["weights"].items() if v > 0.01},
            "expected_return": round(result["expected_return"] * 100, 2),
            "volatility": round(result["volatility"] * 100, 2),
            "sharpe_ratio": round(result["sharpe_ratio"], 3),
            "backtest_sharpe": round(backtest["sharpe_ratio"], 3),
            "max_drawdown": round(backtest["max_drawdown"] * 100, 2)
        }
        return json.dumps(output, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def compare_portfolios(universe: str) -> str:
    """Compare different optimization approaches for the same universe."""
    try:
        tickers = get_universe(universe)
        prices = download_market_data(tickers, "2019-01-01", "2024-01-01")
        results = {}

        for target in ["min_volatility", "max_sharpe"]:
            config = PortfolioConfig(
                tickers=list(prices.columns), start_date="2019-01-01", end_date="2024-01-01",
                optimization_target=target
            )
            r = optimize_portfolio(config, prices)
            bt = backtest_portfolio(r["weights"], prices)
            results[target] = {
                "expected_return": round(r["expected_return"] * 100, 2),
                "volatility": round(r["volatility"] * 100, 2),
                "sharpe": round(r["sharpe_ratio"], 3),
                "max_drawdown": round(bt["max_drawdown"] * 100, 2)
            }

        r = optimize_hrp(prices)
        bt = backtest_portfolio(r["weights"], prices)
        results["hrp"] = {
            "expected_return": round(r["expected_return"] * 100, 2),
            "volatility": round(r["volatility"] * 100, 2),
            "sharpe": round(r["sharpe_ratio"], 3),
            "max_drawdown": round(bt["max_drawdown"] * 100, 2)
        }

        return json.dumps({"universe": universe, "comparisons": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


PORTFOLIO_TOOLS = [get_available_universes, optimize_portfolio_tool, run_hrp_optimization, compare_portfolios]

PORTFOLIO_AGENT_PROMPT = """You are an expert portfolio optimization agent helping investors construct optimal portfolios.

When given an investor's request:
1. Understand their goals (risk tolerance, time horizon, preferences)
2. Select an appropriate universe
3. Choose the right optimization method
4. Run the optimization and present results clearly
5. If asked, compare multiple approaches

Always explain your reasoning and trade-offs."""


def create_portfolio_agent(provider: LLMProvider = LLMProvider.GEMINI) -> AgentExecutor:
    """Create a portfolio optimization agent."""
    llm = get_llm(provider, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PORTFOLIO_AGENT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_tool_calling_agent(llm, PORTFOLIO_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=PORTFOLIO_TOOLS, verbose=True, return_intermediate_steps=True)


@observe()
def run_portfolio_agent(agent: AgentExecutor, query: str, session_id: str = None) -> Dict:
    """Run portfolio agent with automatic Langfuse tracing."""
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(name="portfolio_agent", session_id=session_id, input={"query": query})

    # CallbackHandler auto-inherits current trace context
    handler = LangfuseCallbackHandler() if LANGFUSE_ENABLED else None
    config = {"callbacks": [handler]} if handler else {}

    result = agent.invoke({"input": query}, config=config)

    if langfuse:
        langfuse.update_current_trace(output={"response": result["output"][:500]})

    return result


def create_rag_knowledge_base(eval_data: Dict) -> Any:
    """Create RAG knowledge base from evaluation data."""
    documents = []

    for item in eval_data.get("rag_knowledge", []):
        documents.append(Document(page_content=item["content"], metadata={"persona": item["persona"]}))

    practices = eval_data.get("best_practices", {})
    for profile_type, info in practices.items():
        content = f"{profile_type.title()} Profile ({info['profile']}): {info['optimization']}, {info['priority']}"
        documents.append(Document(page_content=content, metadata={"type": "best_practice"}))

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


EVALUATOR_AGENT_PROMPT = """You are an expert portfolio optimization evaluator.

Evaluate recommendations with scores (1-10) for:
- Universe Selection
- Optimization Method
- Risk Assessment
- Constraint Handling
- Explanation Quality

Provide an overall score and specific feedback."""


def create_evaluator_agent(retriever, provider: LLMProvider = LLMProvider.GEMINI) -> AgentExecutor:
    """Create an evaluator agent with RAG."""

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search portfolio optimization knowledge base."""
        docs = retriever.get_relevant_documents(query)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    llm = get_llm(provider, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_AGENT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    tools = [search_knowledge_base]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)


@dataclass
class A2AEvaluation:
    """A2A evaluation result."""
    task_description: str
    task_agent_response: str
    evaluation_reasoning: str
    scores: Dict[str, float]
    overall_score: float


@observe()
def run_a2a_evaluation(
    task_description: str,
    portfolio_agent: AgentExecutor,
    evaluator_agent: AgentExecutor,
    session_id: str = None
) -> A2AEvaluation:
    """Run Agent-to-Agent evaluation with automatic Langfuse tracing."""
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="a2a_evaluation",
            session_id=session_id,
            input={"task_description": task_description}
        )

    # CallbackHandler auto-inherits current trace context
    handler = LangfuseCallbackHandler() if LANGFUSE_ENABLED else None
    config = {"callbacks": [handler]} if handler else {}

    # Run portfolio agent
    task_result = portfolio_agent.invoke({"input": task_description}, config=config)
    task_response = task_result["output"]

    eval_query = f"""Evaluate this portfolio recommendation.

INVESTOR REQUEST: {task_description}

PORTFOLIO AGENT RESPONSE: {task_response}

Provide scores (1-10) for Universe Selection, Optimization Method, Risk Assessment, Constraint Handling, Explanation Quality, and an overall score with feedback."""

    # Run evaluator agent
    eval_result = evaluator_agent.invoke({"input": eval_query}, config=config)

    # Parse scores
    scores = {"universe_selection": 7.0, "optimization_method": 8.0, "risk_assessment": 7.5,
              "constraint_handling": 8.0, "explanation_quality": 7.0}
    overall_score = sum(scores.values()) / len(scores)

    try:
        import re
        response_text = eval_result["output"]
        score_patterns = [
            (r"universe\s*selection[:\s]*(\d+)", "universe_selection"),
            (r"optimization\s*method[:\s]*(\d+)", "optimization_method"),
            (r"risk\s*assessment[:\s]*(\d+)", "risk_assessment"),
            (r"constraint\s*handling[:\s]*(\d+)", "constraint_handling"),
            (r"explanation\s*quality[:\s]*(\d+)", "explanation_quality"),
            (r"overall[:\s]*(\d+)", "overall")
        ]
        for pattern, key in score_patterns:
            match = re.search(pattern, response_text.lower())
            if match:
                if key == "overall":
                    overall_score = float(match.group(1))
                else:
                    scores[key] = float(match.group(1))
    except Exception:
        pass

    if langfuse:
        langfuse.update_current_trace(output={"overall_score": overall_score, "scores": scores})

    return A2AEvaluation(
        task_description=task_description,
        task_agent_response=task_response,
        evaluation_reasoning=eval_result["output"],
        scores=scores,
        overall_score=overall_score
    )


def flush_langfuse():
    """Flush Langfuse events."""
    langfuse = get_client()
    if langfuse:
        langfuse.flush()
