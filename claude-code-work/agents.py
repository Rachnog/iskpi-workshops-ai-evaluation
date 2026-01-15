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
        docs = retriever.invoke(query)
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


# =============================================================================
# A2A PROTOCOL IMPLEMENTATION
# Green Agent (Evaluator) <-> Purple Agent (Portfolio)
# =============================================================================

@dataclass
class A2AMessage:
    """Message in A2A protocol communication."""
    sender: str  # "green" (evaluator) or "purple" (portfolio)
    content: str
    message_type: str  # "request", "response", "query", "assessment"


@dataclass
class A2AEvaluation:
    """A2A evaluation result with full conversation history."""
    task_description: str
    conversation: list  # List of A2AMessage
    scores: Dict[str, float]
    overall_score: float
    feedback: str


def create_a2a_purple_agent(provider: LLMProvider = LLMProvider.GEMINI) -> AgentExecutor:
    """
    Create Purple Agent (the agent being evaluated).

    Purple agents are competitors that attempt to excel at tasks defined by green agents.
    This is the portfolio optimization agent.
    """
    return create_portfolio_agent(provider)


def create_a2a_green_agent(retriever, provider: LLMProvider = LLMProvider.GEMINI) -> AgentExecutor:
    """
    Create Green Agent (the evaluator).

    Green agents define tasks, environments, and scoring.
    This agent has RAG for knowledge and web search for current info.
    """
    from langchain_community.tools import DuckDuckGoSearchRun

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search portfolio optimization knowledge base for best practices and examples."""
        docs = retriever.invoke(query)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    @tool
    def web_search(query: str) -> str:
        """Search the web for current market information or financial concepts."""
        try:
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            return f"Web search unavailable: {str(e)}"

    GREEN_AGENT_PROMPT = """You are a Green Agent (Evaluator) in an A2A (Agent-to-Agent) evaluation protocol.

Your role:
1. QUERY the Purple Agent (portfolio optimizer) to understand its recommendation
2. ASK follow-up questions if the response is unclear or incomplete
3. SEARCH your knowledge base for best practices and reference examples
4. ASSESS the quality of the recommendation

When evaluating, score these dimensions (1-10):
- Universe Selection: Is the asset universe appropriate?
- Optimization Method: Is the optimization approach suitable?
- Risk Assessment: Is risk properly evaluated?
- Constraint Handling: Are investor constraints respected?
- Explanation Quality: Is the reasoning clear?

After gathering information, provide your final assessment with scores and feedback."""

    llm = get_llm(provider, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", GREEN_AGENT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    tools = [search_knowledge_base, web_search]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)


@observe()
def run_a2a_evaluation(
    task_description: str,
    portfolio_agent: AgentExecutor,
    evaluator_agent: AgentExecutor,
    session_id: str = None,
    max_rounds: int = 3
) -> A2AEvaluation:
    """
    Run Agent-to-Agent evaluation following the A2A protocol.

    Protocol flow (iterative):
    1. Green Agent sends initial request to Purple Agent
    2. Purple Agent responds
    3. Green Agent asks follow-up questions (repeats for max_rounds)
    4. Purple Agent responds to each follow-up
    5. Green Agent produces final assessment with scores

    All communication is logged as A2AMessage objects.
    """
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="a2a_evaluation",
            session_id=session_id,
            input={"task_description": task_description, "max_rounds": max_rounds}
        )

    handler = LangfuseCallbackHandler() if LANGFUSE_ENABLED else None
    config = {"callbacks": [handler]} if handler else {}

    conversation = []
    purple_chat_history = []  # Maintain chat history for Purple agent

    # === ROUND 1: Initial Request ===
    print("\n" + "="*60)
    print("A2A PROTOCOL - ROUND 1: Initial Request")
    print("="*60)

    # Green Agent sends initial request
    initial_request = A2AMessage(
        sender="green",
        content=f"Please provide a portfolio recommendation for this investor: {task_description}",
        message_type="request"
    )
    conversation.append(initial_request)
    print(f"\n[GREEN → PURPLE] {initial_request.content[:200]}...")

    # Purple Agent responds
    purple_result = portfolio_agent.invoke(
        {"input": task_description, "chat_history": purple_chat_history},
        config=config
    )
    purple_response = purple_result["output"]
    purple_chat_history.append({"role": "user", "content": task_description})
    purple_chat_history.append({"role": "assistant", "content": purple_response})

    conversation.append(A2AMessage(
        sender="purple",
        content=purple_response,
        message_type="response"
    ))
    print(f"\n[PURPLE → GREEN] Response received ({len(purple_response)} chars)")

    # === ITERATIVE ROUNDS: Green asks follow-ups, Purple responds ===
    for round_num in range(2, max_rounds + 1):
        print("\n" + "="*60)
        print(f"A2A PROTOCOL - ROUND {round_num}: Follow-up Query")
        print("="*60)

        # Build conversation context for Green agent
        conversation_summary = _format_conversation_for_green(conversation)

        # Green agent formulates a follow-up question
        green_prompt = f"""You are evaluating a portfolio agent. Here is the conversation so far:

INVESTOR REQUEST: {task_description}

CONVERSATION HISTORY:
{conversation_summary}

Based on this conversation, you should:
1. If the portfolio agent's response is incomplete or unclear, ask a specific follow-up question
2. Use your tools (search_knowledge_base, web_search) to gather relevant information
3. Ask the portfolio agent to clarify, expand, or justify their recommendation

Generate a FOLLOW-UP QUESTION to ask the portfolio agent. Be specific and probe deeper into:
- Why they chose that particular universe or optimization method
- How they assessed risk for this investor
- What constraints they considered
- Any alternative approaches they considered

Your question should help you better evaluate the recommendation quality."""

        green_result = evaluator_agent.invoke({"input": green_prompt}, config=config)
        green_question = green_result["output"]

        conversation.append(A2AMessage(
            sender="green",
            content=green_question,
            message_type="query"
        ))
        print(f"\n[GREEN → PURPLE] {green_question[:300]}...")

        # Purple agent responds to the follow-up
        purple_result = portfolio_agent.invoke(
            {"input": green_question, "chat_history": purple_chat_history},
            config=config
        )
        purple_response = purple_result["output"]
        purple_chat_history.append({"role": "user", "content": green_question})
        purple_chat_history.append({"role": "assistant", "content": purple_response})

        conversation.append(A2AMessage(
            sender="purple",
            content=purple_response,
            message_type="response"
        ))
        print(f"\n[PURPLE → GREEN] Response received ({len(purple_response)} chars)")

    # === FINAL ASSESSMENT ===
    print("\n" + "="*60)
    print("A2A PROTOCOL - FINAL ASSESSMENT")
    print("="*60)

    conversation_summary = _format_conversation_for_green(conversation)

    assessment_prompt = f"""You have completed your evaluation of the portfolio agent. Here is the full conversation:

INVESTOR REQUEST: {task_description}

FULL CONVERSATION:
{conversation_summary}

Now provide your FINAL ASSESSMENT. You must provide:
1. Scores (1-10) for each dimension
2. Overall score (1-10)
3. Specific feedback based on the entire conversation

Format your scores EXACTLY like this:
- Universe Selection: X/10
- Optimization Method: X/10
- Risk Assessment: X/10
- Constraint Handling: X/10
- Explanation Quality: X/10
- Overall Score: X/10

Then provide your detailed feedback explaining your scores."""

    final_result = evaluator_agent.invoke({"input": assessment_prompt}, config=config)

    assessment = A2AMessage(
        sender="green",
        content=final_result["output"],
        message_type="assessment"
    )
    conversation.append(assessment)

    # Parse scores from assessment
    scores = _parse_a2a_scores(assessment.content)
    overall_score = scores.get("overall", sum(scores.values()) / max(len(scores), 1))

    print(f"\n[GREEN ASSESSMENT] Overall Score: {overall_score:.1f}/10")

    if langfuse:
        langfuse.update_current_trace(output={
            "overall_score": overall_score,
            "scores": scores,
            "num_rounds": max_rounds
        })

    return A2AEvaluation(
        task_description=task_description,
        conversation=conversation,
        scores=scores,
        overall_score=overall_score,
        feedback=assessment.content
    )


def _format_conversation_for_green(conversation: list) -> str:
    """Format conversation history for the Green agent."""
    lines = []
    for msg in conversation:
        sender = "GREEN (Evaluator)" if msg.sender == "green" else "PURPLE (Portfolio)"
        lines.append(f"[{sender}] ({msg.message_type}):\n{msg.content}\n")
    return "\n---\n".join(lines)


def _parse_a2a_scores(text: str) -> Dict[str, float]:
    """Parse scores from evaluator's assessment text."""
    import re

    scores = {}
    patterns = [
        (r"universe\s*selection[:\s]*(\d+(?:\.\d+)?)", "universe_selection"),
        (r"optimization\s*method[:\s]*(\d+(?:\.\d+)?)", "optimization_method"),
        (r"risk\s*assessment[:\s]*(\d+(?:\.\d+)?)", "risk_assessment"),
        (r"constraint\s*handling[:\s]*(\d+(?:\.\d+)?)", "constraint_handling"),
        (r"explanation\s*quality[:\s]*(\d+(?:\.\d+)?)", "explanation_quality"),
        (r"overall\s*(?:score)?[:\s]*(\d+(?:\.\d+)?)", "overall")
    ]

    for pattern, key in patterns:
        match = re.search(pattern, text.lower())
        if match:
            scores[key] = float(match.group(1))

    # Default scores if parsing fails
    if not scores:
        scores = {
            "universe_selection": 7.0,
            "optimization_method": 7.0,
            "risk_assessment": 7.0,
            "constraint_handling": 7.0,
            "explanation_quality": 7.0,
            "overall": 7.0
        }

    return scores


# Legacy function for backward compatibility
def create_evaluator_agent(retriever, provider: LLMProvider = LLMProvider.GEMINI) -> AgentExecutor:
    """Create an evaluator agent with RAG (legacy - use create_a2a_green_agent for A2A)."""
    return create_a2a_green_agent(retriever, provider)


def flush_langfuse():
    """Flush Langfuse events."""
    langfuse = get_client()
    if langfuse:
        langfuse.flush()
