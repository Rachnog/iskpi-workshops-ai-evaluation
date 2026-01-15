"""
AI Agents Module

Portfolio optimization and evaluation agents using LangChain.
Multi-provider support (Gemini, Anthropic, OpenAI).
Langfuse tracing via @observe() decorator - automatic hierarchical tracing.
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path='../.env')

# Set GOOGLE_API_KEY for langchain-google-genai
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if _gemini_key:
    os.environ["GOOGLE_API_KEY"] = _gemini_key

from langchain.agents import create_tool_calling_agent, create_react_agent, AgentExecutor
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


# =============================================================================
# SKILLS-BASED AGENT IMPLEMENTATION
# =============================================================================

@dataclass
class Skill:
    """Represents a loaded Skill from SKILL.md file."""
    name: str
    description: str
    instructions: str
    path: str

    def __str__(self):
        return f"Skill({self.name}): {self.description[:50]}..."


class SkillLoader:
    """Load and manage Skills from .claude/skills/ directory."""

    def __init__(self, skills_dir: str = None):
        if skills_dir is None:
            skills_dir = os.path.join(os.path.dirname(__file__), ".claude", "skills")
        self.skills_dir = skills_dir
        self.skills: Dict[str, Skill] = {}
        self._load_skills()

    def _load_skills(self):
        """Load all SKILL.md files from the skills directory."""
        import yaml

        if not os.path.exists(self.skills_dir):
            print(f"Skills directory not found: {self.skills_dir}")
            return

        for skill_name in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
            if os.path.exists(skill_path):
                try:
                    with open(skill_path, 'r') as f:
                        content = f.read()

                    # Parse YAML frontmatter
                    if content.startswith('---'):
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            frontmatter = yaml.safe_load(parts[1])
                            instructions = parts[2].strip()

                            self.skills[frontmatter['name']] = Skill(
                                name=frontmatter['name'],
                                description=frontmatter.get('description', ''),
                                instructions=instructions,
                                path=skill_path
                            )
                except Exception as e:
                    print(f"Error loading skill {skill_name}: {e}")

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a specific skill by name."""
        return self.skills.get(name)

    def get_all_skills(self) -> List[Skill]:
        """Get all loaded skills."""
        return list(self.skills.values())

    def get_skill_descriptions(self) -> str:
        """Get formatted descriptions of all skills for the agent prompt."""
        lines = ["## Available Skills\n"]
        for skill in self.skills.values():
            lines.append(f"- **{skill.name}**: {skill.description}")
        return "\n".join(lines)

    def get_skill_instructions(self, skill_name: str) -> str:
        """Get the full instructions for a skill (progressive disclosure)."""
        skill = self.skills.get(skill_name)
        if skill:
            return skill.instructions
        return f"Skill '{skill_name}' not found."


def create_skills_purple_agent(
    provider: LLMProvider = LLMProvider.ANTHROPIC,
    skills_dir: str = None
) -> Tuple[AgentExecutor, SkillLoader]:
    """
    Create a Skills-based Purple Agent (Portfolio Optimizer).

    Uses Anthropic Claude by default, following Claude's Agent Skills pattern.
    Instead of hardcoded tools, this agent uses modular Skills loaded from
    .claude/skills/ directory. Skills provide instructions that the agent
    follows to complete tasks.

    Returns:
        Tuple of (AgentExecutor, SkillLoader)
    """
    from langchain.tools import tool

    # Load skills
    skill_loader = SkillLoader(skills_dir)

    # Persistent execution context - variables persist between calls
    _execution_context = {
        'portfolio_optimizer': None,
        'PortfolioConfig': None,
        'optimize_portfolio': None,
        'optimize_hrp': None,
        'backtest_portfolio': None,
        'download_market_data': None,
        'get_universe': None,
        'UNIVERSE_DEFINITIONS': None,
        'np': None,
        'pd': None,
    }
    _empty_call_count = [0]  # Use list to allow mutation in nested function

    def _init_context():
        """Initialize execution context with portfolio_optimizer functions."""
        if _execution_context['portfolio_optimizer'] is None:
            import portfolio_optimizer as po
            _execution_context.update({
                'portfolio_optimizer': po,
                'PortfolioConfig': po.PortfolioConfig,
                'optimize_portfolio': po.optimize_portfolio,
                'optimize_hrp': po.optimize_hrp,
                'backtest_portfolio': po.backtest_portfolio,
                'download_market_data': po.download_market_data,
                'get_universe': po.get_universe,
                'UNIVERSE_DEFINITIONS': po.UNIVERSE_DEFINITIONS,
                'np': __import__('numpy'),
                'pd': __import__('pandas'),
            })

    # Create tools that invoke skills
    @tool
    def invoke_skill(skill_name: str) -> str:
        """
        Invoke a skill by name to get detailed instructions.
        Available skills: universe-selection, optimization-execution,
        risk-assessment, backtesting, portfolio-comparison
        """
        _empty_call_count[0] = 0  # Reset empty call counter on valid tool use
        return skill_loader.get_skill_instructions(skill_name)

    @tool
    def list_available_skills() -> str:
        """List all available skills and their descriptions."""
        _empty_call_count[0] = 0  # Reset empty call counter on valid tool use
        return skill_loader.get_skill_descriptions()

    @tool
    def execute_portfolio_code(code: str = "") -> str:
        """
        Execute portfolio optimization code. Variables persist between calls.
        The code should use functions from portfolio_optimizer module.
        Pass the Python code as a string to execute.
        """
        if not code or not code.strip():
            _empty_call_count[0] += 1
            if _empty_call_count[0] >= 3:
                return "ERROR: You have called this tool with empty input multiple times. STOP calling execute_portfolio_code and provide your response to the user based on the information you already have."
            return "No code provided. Please provide Python code as a string argument, e.g., execute_portfolio_code(code='print(1+1)')"

        _empty_call_count[0] = 0  # Reset on valid call
        _init_context()

        try:
            # Execute in persistent context - variables survive between calls
            exec(code, _execution_context)

            # Return any result variable if set
            if 'result' in _execution_context:
                result_str = str(_execution_context['result'])
                return result_str if len(result_str) < 2000 else result_str[:2000] + "...[truncated]"
            return "Code executed successfully."
        except Exception as e:
            return f"Error executing code: {str(e)}. Make sure all required variables are defined in this code block."

    tools = [list_available_skills, invoke_skill, execute_portfolio_code]

    # Create agent with Skills-aware prompt
    skills_prompt = skill_loader.get_skill_descriptions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a Skills-based Portfolio Optimization Agent powered by Claude.

You use modular Skills to provide portfolio recommendations. Each Skill contains
specific instructions and code examples for a particular capability.

{skills_prompt}

## How to Use Skills

1. Use `invoke_skill` with a skill name to load its detailed instructions
2. Read the skill instructions carefully - they contain code examples
3. Use `execute_portfolio_code` to run Python code (variables persist between calls)
4. Copy code from skill instructions and adapt it for the specific request

## IMPORTANT: Using execute_portfolio_code

- Variables PERSIST between calls (prices, result, etc. are available in later calls)
- ONLY call when you have specific Python code to execute
- NEVER call with empty input - if you do this 3 times, you must stop and respond
- The code must be a complete Python snippet
- Example: execute_portfolio_code(code="prices = download_market_data(['AAPL'], '2020-01-01', '2024-01-01')")

## Workflow

For portfolio recommendations:
1. Invoke `universe-selection` skill → execute code to get tickers and prices
2. Invoke `optimization-execution` skill → execute code to optimize (prices persists!)
3. Invoke `risk-assessment` skill → execute code for risk metrics (result persists!)
4. Optionally invoke `portfolio-comparison` for alternatives

Since variables persist, you can build on previous results:
- Call 1: `prices = download_market_data(...)` → prices is saved
- Call 2: `result = optimize_portfolio(config, prices)` → uses saved prices
- Call 3: `bt = backtest_portfolio(result['weights'], prices)` → uses both

Do NOT call execute_portfolio_code without providing actual code.
Always explain your reasoning and present results clearly."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    llm = get_llm(provider, temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    ), skill_loader


@observe()
def run_skills_agent(
    agent: AgentExecutor,
    query: str,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Run the Skills-based portfolio agent with Langfuse tracing.
    """
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="skills_portfolio_agent",
            session_id=session_id,
            input={"query": query}
        )

    handler = LangfuseCallbackHandler() if LANGFUSE_ENABLED else None
    config = {"callbacks": [handler]} if handler else {}

    result = agent.invoke({"input": query}, config=config)

    if langfuse:
        langfuse.update_current_trace(output={"result": result["output"][:500]})

    return result


@observe()
def run_skills_a2a_evaluation(
    task_description: str,
    skills_agent: AgentExecutor,
    evaluator_agent: AgentExecutor,
    session_id: str = None,
    max_rounds: int = 3
) -> A2AEvaluation:
    """
    Run A2A evaluation with Skills-based Purple Agent.

    This is similar to run_a2a_evaluation but specifically designed to
    evaluate a Skills-based agent. The Green agent evaluates how well
    the Skills-based Purple agent uses its modular skills.
    """
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="skills_a2a_evaluation",
            session_id=session_id,
            input={"task_description": task_description, "max_rounds": max_rounds}
        )

    handler = LangfuseCallbackHandler() if LANGFUSE_ENABLED else None
    config = {"callbacks": [handler]} if handler else {}

    conversation = []
    purple_chat_history = []

    # === ROUND 1: Initial Request ===
    print("\n" + "="*60)
    print("SKILLS A2A PROTOCOL - ROUND 1: Initial Request")
    print("="*60)

    initial_request = A2AMessage(
        sender="green",
        content=f"Please provide a portfolio recommendation for this investor: {task_description}",
        message_type="request"
    )
    conversation.append(initial_request)
    print(f"\n[GREEN → PURPLE (Skills)] {initial_request.content[:200]}...")

    # Skills-based Purple Agent responds
    purple_result = skills_agent.invoke(
        {"input": task_description},
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
    print(f"\n[PURPLE (Skills) → GREEN] Response received ({len(purple_response)} chars)")

    # === ITERATIVE ROUNDS ===
    for round_num in range(2, max_rounds + 1):
        print("\n" + "="*60)
        print(f"SKILLS A2A PROTOCOL - ROUND {round_num}: Follow-up Query")
        print("="*60)

        conversation_summary = _format_conversation_for_green(conversation)

        green_prompt = f"""You are evaluating a Skills-based portfolio agent. Here is the conversation so far:

INVESTOR REQUEST: {task_description}

CONVERSATION HISTORY:
{conversation_summary}

The Skills-based agent should be using modular Skills (universe-selection, optimization-execution,
risk-assessment, backtesting, portfolio-comparison) to complete tasks.

Based on this conversation:
1. If the response is incomplete, ask for more details
2. Use your tools (search_knowledge_base, web_search) to gather relevant information
3. Probe how well the agent is using its Skills

Generate a FOLLOW-UP QUESTION to ask the Skills-based portfolio agent."""

        green_result = evaluator_agent.invoke({"input": green_prompt}, config=config)
        green_question = green_result["output"]

        conversation.append(A2AMessage(
            sender="green",
            content=green_question,
            message_type="query"
        ))
        print(f"\n[GREEN → PURPLE (Skills)] {green_question[:300]}...")

        # Purple responds to follow-up
        purple_result = skills_agent.invoke(
            {"input": green_question},
            config=config
        )
        purple_response = purple_result["output"]

        conversation.append(A2AMessage(
            sender="purple",
            content=purple_response,
            message_type="response"
        ))
        print(f"\n[PURPLE (Skills) → GREEN] Response received ({len(purple_response)} chars)")

    # === FINAL ASSESSMENT ===
    print("\n" + "="*60)
    print("SKILLS A2A PROTOCOL - FINAL ASSESSMENT")
    print("="*60)

    conversation_summary = _format_conversation_for_green(conversation)

    assessment_prompt = f"""You have completed your evaluation of the Skills-based portfolio agent.

INVESTOR REQUEST: {task_description}

FULL CONVERSATION:
{conversation_summary}

The agent uses a Skills-based architecture with modular capabilities:
- universe-selection: Select appropriate assets
- optimization-execution: Run MVO/HRP optimization
- risk-assessment: Evaluate portfolio risk
- backtesting: Validate with historical data
- portfolio-comparison: Compare alternatives

Evaluate how well the agent:
1. Used appropriate Skills for the task
2. Followed Skill instructions correctly
3. Provided a complete recommendation

Provide your FINAL ASSESSMENT with scores:
- Universe Selection: X/10
- Optimization Method: X/10
- Risk Assessment: X/10
- Constraint Handling: X/10
- Explanation Quality: X/10
- Skills Usage: X/10 (how well it used the modular Skills)
- Overall Score: X/10

Then provide detailed feedback."""

    final_result = evaluator_agent.invoke({"input": assessment_prompt}, config=config)

    assessment = A2AMessage(
        sender="green",
        content=final_result["output"],
        message_type="assessment"
    )
    conversation.append(assessment)

    scores = _parse_a2a_scores(assessment.content)
    overall_score = scores.get("overall", sum(scores.values()) / max(len(scores), 1))

    print(f"\n[GREEN ASSESSMENT] Overall Score: {overall_score:.1f}/10")

    if langfuse:
        langfuse.update_current_trace(output={
            "overall_score": overall_score,
            "scores": scores,
            "num_rounds": max_rounds,
            "agent_type": "skills-based"
        })

    return A2AEvaluation(
        task_description=task_description,
        conversation=conversation,
        scores=scores,
        overall_score=overall_score,
        feedback=assessment.content
    )


# =============================================================================
# NATIVE ANTHROPIC SKILLS IMPLEMENTATION
# Uses Anthropic SDK directly with bash/text_editor tools
# =============================================================================

def create_native_skills_agent(skills_dir: str = None) -> Dict[str, Any]:
    """
    Create a native Anthropic Skills-based agent.

    This uses the Anthropic SDK directly with:
    - bash_20250124 tool for reading skill files
    - text_editor_20250124 tool for file operations
    - Skills read from filesystem as per Anthropic's Skills architecture

    Returns:
        Dict with 'client', 'skill_loader', 'betas', 'model'
    """
    import anthropic

    # Initialize client
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=anthropic_key)

    # Load skill metadata
    skill_loader = SkillLoader(skills_dir)

    return {
        "client": client,
        "skill_loader": skill_loader,
        "model": "claude-sonnet-4-20250514"
    }


@observe()
def run_native_skills_agent(
    agent_config: Dict[str, Any],
    query: str,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Run the native Anthropic Skills agent.

    Claude uses bash tool to:
    1. Read skill files from the filesystem (cat SKILL.md)
    2. Execute Python code for portfolio optimization
    3. Return results with persistent state in conversation
    """
    client = agent_config["client"]
    skill_loader = agent_config["skill_loader"]

    # Build system prompt with skill metadata
    skill_descriptions = skill_loader.get_skill_descriptions()
    skills_path = skill_loader.skills_dir

    # Get the working directory (where portfolio_optimizer.py is)
    work_dir = os.path.dirname(skills_path)  # This is claude-code-work directory

    system_prompt = f"""You are a Portfolio Optimization Agent using the Skills architecture.

{skill_descriptions}

## How to Use Skills

Skills are located at: {skills_path}

### Workflow (3 steps max):

**Step 1**: Read the optimization-execution skill to get the ready-to-run script:
```bash
cat {skills_path}/optimization-execution/SKILL.md
```

**Step 2**: Run the script from the skill, modifying UNIVERSE, TARGET, and MAX_POSITION for the investor:
```bash
cd {work_dir} && python3 << 'PYEOF'
# Copy the "Complete Ready-to-Run Script" from the skill
# Modify: UNIVERSE, TARGET, MAX_POSITION based on investor profile
PYEOF
```

**Step 3**: Present results clearly to the investor with:
- Recommended allocation with percentages
- Expected return and risk metrics
- Why this portfolio fits their profile

## Quick Reference (use if you don't need to read skill files)

| Investor Type | UNIVERSE | TARGET |
|--------------|----------|--------|
| Conservative (low risk, near retirement) | conservative | min_volatility |
| Balanced (moderate risk, diversified) | global_diversified | max_sharpe |
| Aggressive (high risk, growth) | us_tech | max_sharpe |

## Constraint Handling

If the investor specifies a max position limit (e.g., "no more than 15% in any single position"):
- Set MAX_POSITION = 0.15 in the script

## Important Guidelines

- Complete the task in 3 tool calls or fewer
- Read skill files ONLY if you need detailed guidance
- Present clear, actionable recommendations
- Explain why the portfolio fits the investor's needs"""

    try:
        response = client.messages.create(
            model=agent_config["model"],
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
            tools=[
                {"type": "bash_20250124", "name": "bash"}
            ]
        )

        # Handle tool use in a loop
        messages = [{"role": "user", "content": query}]
        max_iterations = 10  # Reduced - agent should be efficient

        for iteration in range(max_iterations):
            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                # Add assistant's response
                messages.append({"role": "assistant", "content": response.content})

                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [Tool: {block.name}]")
                        tool_result = _execute_native_tool(block, skills_path)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result[:8000]  # Limit size
                        })

                messages.append({"role": "user", "content": tool_results})

                response = client.messages.create(
                    model=agent_config["model"],
                    max_tokens=8192,
                    system=system_prompt,
                    messages=messages,
                    tools=[
                        {"type": "bash_20250124", "name": "bash"}
                    ]
                )

        # Extract final text response
        output = ""
        for block in response.content:
            if hasattr(block, "text"):
                output += block.text

        # If we hit max iterations without end_turn, add note
        if response.stop_reason != "end_turn":
            output += "\n\n[Note: Agent reached iteration limit. Results may be incomplete.]"

        return {
            "output": output,
            "messages": messages,
            "stop_reason": response.stop_reason,
            "iterations": iteration + 1
        }

    except Exception as e:
        import traceback
        return {
            "output": f"Error: {str(e)}",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def _execute_native_tool(tool_block, skills_path: str) -> str:
    """Execute a tool call from the native agent."""
    tool_name = tool_block.name
    tool_input = tool_block.input

    # Working directory is the parent of skills_path (claude-code-work)
    work_dir = os.path.dirname(skills_path) if skills_path else os.getcwd()

    if tool_name == "bash":
        command = tool_input.get("command", "")
        try:
            import subprocess
            # Execute in the working directory (where portfolio_optimizer.py is)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout for data download
                cwd=work_dir
            )
            output = result.stdout
            if result.stderr:
                # Filter out common yfinance warnings
                stderr_lines = [l for l in result.stderr.split('\n')
                               if l and 'FutureWarning' not in l and 'UserWarning' not in l]
                if stderr_lines:
                    output += f"\nSTDERR: {chr(10).join(stderr_lines[:10])}"  # Limit stderr
            if result.returncode != 0 and not output:
                output = f"Command failed with return code: {result.returncode}"
            return output if output else "Command completed (no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out after 120 seconds. Try a simpler query."
        except Exception as e:
            return f"Error: {str(e)}"

    elif tool_name == "str_replace_editor":
        # Handle text editor commands (view, create, str_replace)
        command = tool_input.get("command", "")
        path = tool_input.get("path", "")

        if command == "view":
            try:
                # If path is relative, resolve from work_dir
                if not os.path.isabs(path):
                    path = os.path.join(work_dir, path)
                with open(path, 'r') as f:
                    content = f.read()
                return content[:10000]  # Limit output
            except Exception as e:
                return f"Error reading file: {str(e)}"

        return f"Text editor command '{command}' executed"

    return f"Unknown tool: {tool_name}"


@observe()
def run_native_skills_a2a_evaluation(
    task_description: str,
    native_agent_config: Dict[str, Any],
    evaluator_agent: AgentExecutor,
    session_id: str = None,
    max_rounds: int = 3
) -> A2AEvaluation:
    """
    Run A2A evaluation with Native Anthropic Skills agent.

    The Purple agent uses native Anthropic SDK with bash tool.
    The Green agent remains a LangChain agent for evaluation.
    """
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="native_skills_a2a_evaluation",
            session_id=session_id,
            input={"task_description": task_description, "max_rounds": max_rounds}
        )

    handler = LangfuseCallbackHandler() if LANGFUSE_ENABLED else None
    config = {"callbacks": [handler]} if handler else {}

    conversation = []

    # === ROUND 1: Initial Request ===
    print("\n" + "="*60)
    print("NATIVE SKILLS A2A PROTOCOL - ROUND 1: Initial Request")
    print("="*60)

    initial_request = A2AMessage(
        sender="green",
        content=f"Please provide a portfolio recommendation for this investor: {task_description}",
        message_type="request"
    )
    conversation.append(initial_request)
    print(f"\n[GREEN → PURPLE (Native Skills)] {initial_request.content[:200]}...")

    # Native Skills agent responds
    purple_result = run_native_skills_agent(native_agent_config, task_description, session_id)
    purple_response = purple_result.get("output", "Error: No response")

    conversation.append(A2AMessage(
        sender="purple",
        content=purple_response,
        message_type="response"
    ))
    print(f"\n[PURPLE (Native Skills) → GREEN] Response received ({len(purple_response)} chars)")

    # === ITERATIVE ROUNDS ===
    for round_num in range(2, max_rounds + 1):
        print("\n" + "="*60)
        print(f"NATIVE SKILLS A2A PROTOCOL - ROUND {round_num}: Follow-up Query")
        print("="*60)

        conversation_summary = _format_conversation_for_green(conversation)

        green_prompt = f"""You are evaluating a Native Skills-based portfolio agent.

INVESTOR REQUEST: {task_description}

CONVERSATION HISTORY:
{conversation_summary}

The agent uses Native Anthropic Skills architecture:
- Reads skill SKILL.md files via bash cat command
- Executes Python code via bash python3 -c command
- Skills define workflows for: universe-selection, optimization, risk-assessment, backtesting

Generate a FOLLOW-UP QUESTION to probe the quality of the recommendation."""

        green_result = evaluator_agent.invoke({"input": green_prompt}, config=config)
        green_question = green_result["output"]

        conversation.append(A2AMessage(
            sender="green",
            content=green_question,
            message_type="query"
        ))
        print(f"\n[GREEN → PURPLE (Native Skills)] {green_question[:300]}...")

        # Purple responds to follow-up
        purple_result = run_native_skills_agent(native_agent_config, green_question, session_id)
        purple_response = purple_result.get("output", "Error: No response")

        conversation.append(A2AMessage(
            sender="purple",
            content=purple_response,
            message_type="response"
        ))
        print(f"\n[PURPLE (Native Skills) → GREEN] Response received ({len(purple_response)} chars)")

    # === FINAL ASSESSMENT ===
    print("\n" + "="*60)
    print("NATIVE SKILLS A2A PROTOCOL - FINAL ASSESSMENT")
    print("="*60)

    conversation_summary = _format_conversation_for_green(conversation)

    assessment_prompt = f"""You have completed your evaluation of the Native Skills-based portfolio agent.

INVESTOR REQUEST: {task_description}

FULL CONVERSATION:
{conversation_summary}

The agent uses Native Anthropic Skills architecture:
- Reads skill files via bash commands
- Executes Python code via bash
- Skills: universe-selection, optimization-execution, risk-assessment, backtesting, portfolio-comparison

Provide your FINAL ASSESSMENT with scores:
- Universe Selection: X/10
- Optimization Method: X/10
- Risk Assessment: X/10
- Constraint Handling: X/10
- Explanation Quality: X/10
- Skills Usage: X/10 (how well it used the Skills architecture)
- Overall Score: X/10

Then provide detailed feedback."""

    final_result = evaluator_agent.invoke({"input": assessment_prompt}, config=config)

    assessment = A2AMessage(
        sender="green",
        content=final_result["output"],
        message_type="assessment"
    )
    conversation.append(assessment)

    scores = _parse_a2a_scores(assessment.content)
    overall_score = scores.get("overall", sum(scores.values()) / max(len(scores), 1))

    print(f"\n[GREEN ASSESSMENT] Overall Score: {overall_score:.1f}/10")

    if langfuse:
        langfuse.update_current_trace(output={
            "overall_score": overall_score,
            "scores": scores,
            "num_rounds": max_rounds,
            "agent_type": "native-skills"
        })

    return A2AEvaluation(
        task_description=task_description,
        conversation=conversation,
        scores=scores,
        overall_score=overall_score,
        feedback=assessment.content
    )
