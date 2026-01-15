"""
LLM Translation and Evaluation Utilities

Translation and evaluation are completely decoupled:
- Translation: converts narratives to portfolio configs
- Evaluation: assesses translation quality (field accuracy + LLM-as-judge)

All functions are traced to Langfuse via @observe() decorator.
"""

import os
import json
from typing import Dict, Any, Optional, Literal, List
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path='../.env')

# Set GOOGLE_API_KEY for langchain-google-genai
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if _gemini_key:
    os.environ["GOOGLE_API_KEY"] = _gemini_key

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Langfuse tracing
try:
    from langfuse import observe, get_client
    LANGFUSE_ENABLED = True
except ImportError:
    LANGFUSE_ENABLED = False
    observe = lambda *a, **kw: (lambda f: f)  # no-op decorator
    get_client = lambda: None


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


# =============================================================================
# SCHEMA
# =============================================================================

class PortfolioConfigSchema(BaseModel):
    """Schema for portfolio optimization configuration."""
    optimization_target: Literal["min_volatility", "max_sharpe", "max_return", "efficient_return"] = Field(
        description="Optimization objective: min_volatility for low risk, max_sharpe for balanced, max_return for aggressive"
    )
    universe: Literal["us_large_cap", "us_tech", "global_diversified", "european", "asian", "conservative", "aggressive"] = Field(
        description="Asset universe to invest in"
    )
    time_horizon_years: int = Field(
        description="Investment horizon in years",
        ge=1, le=50
    )
    max_position: Optional[float] = Field(
        default=None,
        description="Maximum weight per position (0.05-1.0), null if not specified",
        ge=0.05, le=1.0
    )
    risk_tolerance: Literal["low", "medium", "high"] = Field(
        description="Risk tolerance level inferred from narrative"
    )
    allow_short: bool = Field(
        default=False,
        description="Whether short selling is allowed"
    )
    target_return: Optional[float] = Field(
        default=None,
        description="Target return for efficient_return optimization, null otherwise"
    )
    reasoning: str = Field(
        description="Brief explanation of why these parameters were chosen"
    )


# =============================================================================
# TRANSLATION (standalone - no evaluation logic)
# =============================================================================

TRANSLATION_PROMPT = """You are an expert financial advisor. Translate the investor narrative into a structured portfolio configuration.

INVESTOR NARRATIVE:
{narrative}

AVAILABLE OPTIONS:
- optimization_target: "min_volatility" (conservative), "max_sharpe" (balanced), "max_return" (aggressive), "efficient_return" (target return)
- universe: "us_large_cap", "us_tech", "global_diversified", "european", "asian", "conservative", "aggressive"
- risk_tolerance: "low", "medium", "high"

Return ONLY a valid JSON object with these exact fields:
- optimization_target (string from options above)
- universe (string from options above)
- time_horizon_years (integer, default 10 if not specified)
- max_position (float 0.05-1.0, or null if not specified)
- risk_tolerance (string from options above)
- allow_short (boolean, default false)
- target_return (float or null)
- reasoning (brief explanation string)

Do not include any text before or after the JSON."""


@observe()
def translate_narrative(
    narrative: str,
    provider: LLMProvider = LLMProvider.GEMINI,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Translate investor narrative to portfolio configuration.

    This is a pure translation function - no evaluation logic.
    Traced to Langfuse automatically via @observe().
    """
    import re

    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="translate_narrative",
            session_id=session_id,
            input={"narrative": narrative, "provider": provider.value}
        )

    llm = get_llm(provider, temperature=0)
    prompt = TRANSLATION_PROMPT.format(narrative=narrative)

    try:
        response = llm.invoke(prompt)
        content = response.content

        # Extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            raise ValueError("No JSON object found in response")

        raw_config = json.loads(json_match.group())

        # Validate with Pydantic schema
        validated = PortfolioConfigSchema(**raw_config)
        config = validated.model_dump()

        if langfuse:
            langfuse.update_current_trace(output={"config": config, "status": "success"})

        return {"config": config, "status": "success"}
    except Exception as e:
        error_result = {"error": str(e), "status": "failed"}
        if langfuse:
            langfuse.update_current_trace(output=error_result)
        return error_result


# =============================================================================
# EVALUATION (standalone - completely decoupled from translation)
# =============================================================================

@observe()
def compute_field_accuracy(
    predicted: Dict,
    expected: Dict,
    session_id: str = None
) -> Dict[str, float]:
    """
    Compute field-by-field accuracy between predicted and expected configs.

    Completely decoupled from translation - just compares two dicts.
    Traced to Langfuse automatically via @observe().

    Args:
        predicted: The predicted configuration dict
        expected: The expected/ground truth configuration dict
        session_id: Optional Langfuse session ID

    Returns:
        Dict with field match scores and overall_accuracy
    """
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="compute_field_accuracy",
            session_id=session_id,
            input={"predicted_keys": list(predicted.keys()), "expected_keys": list(expected.keys())}
        )

    results = {}

    # Critical fields - exact match required
    critical_fields = ["optimization_target", "universe", "risk_tolerance", "allow_short"]
    for field in critical_fields:
        if field in expected:
            results[f"{field}_match"] = 1.0 if predicted.get(field) == expected.get(field) else 0.0

    # Numeric fields - tolerance-based match (20% tolerance)
    numeric_fields = ["time_horizon_years", "max_position", "target_return"]
    for field in numeric_fields:
        if field in expected and expected[field] is not None:
            pred_val = predicted.get(field)
            exp_val = expected[field]
            if pred_val is not None:
                tolerance = 0.2 * abs(exp_val) if exp_val != 0 else 0.1
                results[f"{field}_match"] = 1.0 if abs(pred_val - exp_val) <= tolerance else 0.0
            else:
                results[f"{field}_match"] = 0.0

    results["overall_accuracy"] = sum(results.values()) / len(results) if results else 0.0

    if langfuse:
        langfuse.update_current_trace(output=results)

    return results


@observe()
def llm_as_judge(
    narrative: str,
    predicted: Dict,
    expected: Dict,
    provider: LLMProvider = LLMProvider.GEMINI,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to evaluate translation quality.

    Completely decoupled from translation - evaluates pre-computed predictions.
    Traced to Langfuse automatically via @observe().

    Args:
        narrative: Original investor narrative
        predicted: The predicted configuration dict
        expected: The expected/ground truth configuration dict
        provider: LLM provider for the judge
        session_id: Optional Langfuse session ID

    Returns:
        Dict with dimension scores and feedback
    """
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="llm_as_judge",
            session_id=session_id,
            input={"narrative_length": len(narrative), "provider": provider.value}
        )

    llm = get_llm(provider, temperature=0)

    judge_prompt = f"""You are evaluating a translation from investor narrative to portfolio configuration.

ORIGINAL NARRATIVE:
{narrative}

PREDICTED CONFIG:
{json.dumps(predicted, indent=2)}

EXPECTED CONFIG:
{json.dumps(expected, indent=2)}

Rate the translation on these dimensions (1-10 scale):
1. optimization_target_score: How well does the predicted target match the investor's goals?
2. universe_score: How appropriate is the asset universe selection?
3. risk_assessment_score: How accurately was risk tolerance assessed?
4. constraints_score: Were constraints (max_position, allow_short, time_horizon) handled correctly?

Respond with ONLY a valid JSON object:
{{"optimization_target_score": <int>, "universe_score": <int>, "risk_assessment_score": <int>, "constraints_score": <int>, "overall_score": <float>, "feedback": "<brief feedback>"}}"""

    try:
        response = llm.invoke(judge_prompt)
        content = response.content

        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            # Calculate overall if not provided
            if "overall_score" not in scores:
                numeric_scores = [v for k, v in scores.items() if k.endswith("_score") and isinstance(v, (int, float))]
                scores["overall_score"] = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
        else:
            scores = {"error": "Could not parse JSON from response", "raw_response": content[:500]}
    except Exception as e:
        scores = {"error": str(e)}

    if langfuse:
        langfuse.update_current_trace(output=scores)

    return scores


@observe()
def evaluate_single(
    narrative: str,
    predicted: Dict,
    expected: Dict,
    provider: LLMProvider = LLMProvider.GEMINI,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Run full evaluation on a single translation.

    Combines field accuracy and LLM-as-judge evaluation.
    Both are traced separately to Langfuse.

    Args:
        narrative: Original investor narrative
        predicted: The predicted configuration dict
        expected: The expected/ground truth configuration dict
        provider: LLM provider for the judge
        session_id: Optional Langfuse session ID

    Returns:
        Dict with field_accuracy, llm_judge_scores, and overall metrics
    """
    langfuse = get_client()
    if langfuse:
        langfuse.update_current_trace(
            name="evaluate_single",
            session_id=session_id,
            input={"has_predicted": predicted is not None, "has_expected": expected is not None}
        )

    # Run field accuracy (traced separately)
    field_accuracy = compute_field_accuracy(predicted, expected, session_id=session_id)

    # Run LLM-as-judge (traced separately)
    llm_scores = llm_as_judge(narrative, predicted, expected, provider=provider, session_id=session_id)

    result = {
        "field_accuracy": field_accuracy,
        "llm_judge_scores": llm_scores,
        "overall_field_accuracy": field_accuracy.get("overall_accuracy", 0),
        "overall_llm_score": llm_scores.get("overall_score", 0) if "error" not in llm_scores else 0
    }

    if langfuse:
        langfuse.update_current_trace(output={
            "overall_field_accuracy": result["overall_field_accuracy"],
            "overall_llm_score": result["overall_llm_score"]
        })

    return result


# =============================================================================
# LANGFUSE EXPERIMENTS (following official SDK pattern)
# =============================================================================

# Import Evaluation class for evaluators
try:
    from langfuse import Evaluation
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    Evaluation = None


def create_translation_task(provider: LLMProvider = LLMProvider.GEMINI):
    """
    Create a task function for translation experiments.

    Returns a function compatible with langfuse.run_experiment().
    """
    def translation_task(*, item, **kwargs):
        """Translate narrative to portfolio config."""
        # Handle both dict items (local data) and dataset items
        if hasattr(item, 'input'):
            narrative = item.input.get("narrative") or item.input.get("text")
        else:
            narrative = item.get("input", {}).get("narrative") or item.get("narrative")

        result = translate_narrative(narrative, provider=provider)

        if result["status"] == "success":
            return result["config"]
        else:
            return {"error": result.get("error"), "status": "failed"}

    return translation_task


def field_accuracy_evaluator(*, output, expected_output, **kwargs):
    """
    Evaluator: Compare predicted vs expected config fields.

    Returns Evaluation with field accuracy score (0-1).
    """
    if not EVALUATION_AVAILABLE:
        return None

    if not output or "error" in output:
        return Evaluation(name="field_accuracy", value=0.0, comment="Translation failed")

    # Compute field-by-field accuracy
    results = {}
    critical_fields = ["optimization_target", "universe", "risk_tolerance", "allow_short"]
    for field in critical_fields:
        if field in expected_output:
            results[field] = 1.0 if output.get(field) == expected_output.get(field) else 0.0

    numeric_fields = ["time_horizon_years", "max_position", "target_return"]
    for field in numeric_fields:
        if field in expected_output and expected_output[field] is not None:
            pred_val = output.get(field)
            exp_val = expected_output[field]
            if pred_val is not None:
                tolerance = 0.2 * abs(exp_val) if exp_val != 0 else 0.1
                results[field] = 1.0 if abs(pred_val - exp_val) <= tolerance else 0.0
            else:
                results[field] = 0.0

    accuracy = sum(results.values()) / len(results) if results else 0.0

    # Build comment showing which fields matched
    matches = [f for f, v in results.items() if v == 1.0]
    misses = [f for f, v in results.items() if v == 0.0]
    comment = f"Matched: {matches}, Missed: {misses}"

    return Evaluation(name="field_accuracy", value=accuracy, comment=comment)


def create_llm_judge_evaluator(provider: LLMProvider = LLMProvider.GEMINI):
    """
    Create an LLM-as-judge evaluator function.

    Returns an evaluator compatible with langfuse.run_experiment().
    """
    def llm_judge_evaluator(*, input, output, expected_output, **kwargs):
        """Use LLM to judge translation quality."""
        if not EVALUATION_AVAILABLE:
            return None

        if not output or "error" in output:
            return Evaluation(name="llm_judge", value=0.0, comment="Translation failed")

        # Get narrative from input
        if hasattr(input, 'get'):
            narrative = input.get("narrative") or input.get("text", "")
        else:
            narrative = str(input)

        # Call LLM judge
        scores = llm_as_judge(narrative, output, expected_output, provider=provider)

        overall = scores.get("overall_score", 0) / 10.0  # Normalize to 0-1
        feedback = scores.get("feedback", "")

        return Evaluation(name="llm_judge", value=overall, comment=feedback)

    return llm_judge_evaluator


def avg_accuracy_evaluator(*, item_results, **kwargs):
    """
    Run-level evaluator: Calculate average field accuracy across all items.
    """
    if not EVALUATION_AVAILABLE:
        return None

    accuracies = [
        eval.value for result in item_results
        for eval in (result.evaluations or [])
        if eval.name == "field_accuracy" and eval.value is not None
    ]

    if not accuracies:
        return Evaluation(name="avg_field_accuracy", value=None, comment="No accuracy scores")

    avg = sum(accuracies) / len(accuracies)
    return Evaluation(
        name="avg_field_accuracy",
        value=avg,
        comment=f"Average across {len(accuracies)} items: {avg:.2%}"
    )


def avg_llm_score_evaluator(*, item_results, **kwargs):
    """
    Run-level evaluator: Calculate average LLM judge score across all items.
    """
    if not EVALUATION_AVAILABLE:
        return None

    scores = [
        eval.value for result in item_results
        for eval in (result.evaluations or [])
        if eval.name == "llm_judge" and eval.value is not None
    ]

    if not scores:
        return Evaluation(name="avg_llm_judge", value=None, comment="No LLM scores")

    avg = sum(scores) / len(scores)
    return Evaluation(
        name="avg_llm_judge",
        value=avg,
        comment=f"Average across {len(scores)} items: {avg:.1%}"
    )


def run_translation_experiment(
    experiment_name: str,
    data: List[Dict[str, Any]],
    provider: LLMProvider = LLMProvider.GEMINI,
    description: str = None,
    include_llm_judge: bool = True
):
    """
    Run a translation experiment using Langfuse's run_experiment() API.

    Args:
        experiment_name: Name for this experiment
        data: List of dicts with 'input' (narrative) and 'expected_output' (config)
        provider: LLM provider to use
        description: Optional experiment description
        include_llm_judge: Whether to include LLM-as-judge evaluation

    Returns:
        Experiment result object from Langfuse
    """
    langfuse = get_client()
    if not langfuse:
        raise RuntimeError("Langfuse client not available")

    # Create task and evaluators
    task = create_translation_task(provider)

    evaluators = [field_accuracy_evaluator]
    if include_llm_judge:
        evaluators.append(create_llm_judge_evaluator(provider))

    run_evaluators = [avg_accuracy_evaluator]
    if include_llm_judge:
        run_evaluators.append(avg_llm_score_evaluator)

    # Run experiment
    print(f"Running experiment: {experiment_name}")
    print(f"Data items: {len(data)}")
    print(f"Provider: {provider.value}")
    print(f"Evaluators: field_accuracy" + (", llm_judge" if include_llm_judge else ""))
    print()

    result = langfuse.run_experiment(
        name=experiment_name,
        description=description or f"Portfolio translation evaluation with {provider.value}",
        data=data,
        task=task,
        evaluators=evaluators,
        run_evaluators=run_evaluators
    )

    return result


def run_dataset_experiment(
    dataset_name: str,
    run_name: str,
    provider: LLMProvider = LLMProvider.GEMINI,
    description: str = None,
    include_llm_judge: bool = True
):
    """
    Run experiment on a Langfuse dataset using dataset.run_experiment() API.

    Args:
        dataset_name: Name of the Langfuse dataset
        run_name: Name for this experiment run
        provider: LLM provider to use
        description: Optional run description
        include_llm_judge: Whether to include LLM-as-judge evaluation

    Returns:
        Experiment result object from Langfuse
    """
    langfuse = get_client()
    if not langfuse:
        raise RuntimeError("Langfuse client not available")

    # Get dataset
    dataset = langfuse.get_dataset(name=dataset_name)

    # Create task and evaluators
    task = create_translation_task(provider)

    evaluators = [field_accuracy_evaluator]
    if include_llm_judge:
        evaluators.append(create_llm_judge_evaluator(provider))

    run_evaluators = [avg_accuracy_evaluator]
    if include_llm_judge:
        run_evaluators.append(avg_llm_score_evaluator)

    print(f"Running experiment: {run_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Provider: {provider.value}")
    print(f"Evaluators: field_accuracy" + (", llm_judge" if include_llm_judge else ""))
    print()

    result = dataset.run_experiment(
        name=run_name,
        description=description or f"Evaluation run with {provider.value}",
        task=task,
        evaluators=evaluators,
        run_evaluators=run_evaluators
    )

    return result


def create_langfuse_dataset(
    dataset_name: str,
    items: List[Dict[str, Any]],
    description: str = None
):
    """
    Create a Langfuse dataset with evaluation items.

    Args:
        dataset_name: Name for the dataset
        items: List of dicts with 'input' and 'expected_output'
        description: Optional description

    Returns:
        Langfuse dataset object
    """
    langfuse = get_client()
    if not langfuse:
        raise RuntimeError("Langfuse client not available")

    # Create dataset
    dataset = langfuse.create_dataset(
        name=dataset_name,
        description=description or "Portfolio translation evaluation dataset"
    )

    # Add items
    for i, item in enumerate(items):
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item.get("input", {"narrative": item.get("narrative")}),
            expected_output=item.get("expected_output", item.get("expected")),
            metadata={"index": i}
        )
        print(f"  Added item {i+1}/{len(items)}")

    langfuse.flush()
    print(f"\nDataset '{dataset_name}' created with {len(items)} items")
    return dataset


# =============================================================================
# UTILITY
# =============================================================================

def flush_langfuse():
    """Flush Langfuse events to ensure they're sent."""
    langfuse = get_client()
    if langfuse:
        langfuse.flush()
