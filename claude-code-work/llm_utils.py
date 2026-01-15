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
# LANGFUSE NATIVE DATASET AND SCORING
# =============================================================================

def create_langfuse_dataset(
    dataset_name: str,
    items: List[Dict[str, Any]],
    description: str = None
) -> Any:
    """
    Create or update a Langfuse dataset with evaluation items.

    Args:
        dataset_name: Name for the dataset in Langfuse
        items: List of dicts with 'input' (narrative) and 'expected_output' (config)
        description: Optional dataset description

    Returns:
        Langfuse dataset object
    """
    langfuse = get_client()
    if not langfuse:
        raise RuntimeError("Langfuse client not available")

    # Create or get the dataset
    dataset = langfuse.create_dataset(
        name=dataset_name,
        description=description or "Portfolio translation evaluation dataset"
    )

    # Add items to the dataset
    for i, item in enumerate(items):
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item.get("input", {"narrative": item.get("narrative")}),
            expected_output=item.get("expected_output", item.get("expected")),
            metadata={"index": i}
        )
        print(f"  Added item {i+1}/{len(items)} to dataset")

    langfuse.flush()
    print(f"\nDataset '{dataset_name}' created with {len(items)} items")
    return dataset


def run_dataset_experiment(
    dataset_name: str,
    run_name: str,
    provider: LLMProvider = LLMProvider.GEMINI,
    run_description: str = None
) -> Dict[str, Any]:
    """
    Run translation + evaluation experiment on a Langfuse dataset.

    Uses Langfuse native features:
    - item.run() context manager for automatic trace linking
    - root_span.score_trace() for native scoring

    Args:
        dataset_name: Name of the Langfuse dataset
        run_name: Name for this experiment run
        provider: LLM provider to use
        run_description: Optional run description

    Returns:
        Dict with experiment results and statistics
    """
    langfuse = get_client()
    if not langfuse:
        raise RuntimeError("Langfuse client not available")

    # Fetch the dataset
    dataset = langfuse.get_dataset(name=dataset_name)
    print(f"Running experiment '{run_name}' on dataset '{dataset_name}'")
    print(f"Dataset has {len(dataset.items)} items\n")

    results = []

    for i, item in enumerate(dataset.items):
        print(f"Processing item {i+1}/{len(dataset.items)}...")

        # Use native Langfuse context manager for automatic trace linking
        with item.run(
            run_name=run_name,
            run_description=run_description or f"Translation evaluation run",
            run_metadata={"provider": provider.value, "index": i}
        ) as root_span:

            narrative = item.input.get("narrative") or item.input.get("text")
            expected = item.expected_output

            # Step 1: Translate
            translation = translate_narrative(narrative, provider=provider)

            if translation["status"] != "success":
                print(f"  Translation failed: {translation.get('error')}")
                root_span.score_trace(name="translation_success", value=0)
                results.append({"index": i, "status": "failed", "error": translation.get("error")})
                continue

            predicted = translation["config"]

            # Step 2: Compute field accuracy
            field_accuracy = compute_field_accuracy(predicted, expected)
            overall_accuracy = field_accuracy["overall_accuracy"]

            # Step 3: LLM-as-judge evaluation
            llm_scores = llm_as_judge(narrative, predicted, expected, provider=provider)
            overall_llm_score = llm_scores.get("overall_score", 0)

            # Score the trace using Langfuse native scoring
            root_span.score_trace(
                name="field_accuracy",
                value=overall_accuracy,
                comment=f"Fields: {json.dumps({k: v for k, v in field_accuracy.items() if k != 'overall_accuracy'})}"
            )

            root_span.score_trace(
                name="llm_judge_score",
                value=overall_llm_score / 10.0,  # Normalize to 0-1
                comment=llm_scores.get("feedback", "")
            )

            # Individual dimension scores
            for dim in ["optimization_target_score", "universe_score", "risk_assessment_score", "constraints_score"]:
                if dim in llm_scores:
                    root_span.score_trace(
                        name=dim,
                        value=llm_scores[dim] / 10.0
                    )

            results.append({
                "index": i,
                "status": "success",
                "field_accuracy": overall_accuracy,
                "llm_score": overall_llm_score,
                "predicted": predicted
            })

            print(f"  Field accuracy: {overall_accuracy:.2%}, LLM score: {overall_llm_score:.1f}/10")

    # Flush to ensure all scores are sent
    langfuse.flush()

    # Compute summary statistics
    successful = [r for r in results if r["status"] == "success"]
    avg_field_accuracy = sum(r["field_accuracy"] for r in successful) / len(successful) if successful else 0
    avg_llm_score = sum(r["llm_score"] for r in successful) / len(successful) if successful else 0

    summary = {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "total_items": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_field_accuracy": avg_field_accuracy,
        "avg_llm_score": avg_llm_score,
        "detailed_results": results
    }

    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY: {run_name}")
    print(f"{'='*60}")
    print(f"Total items: {summary['total_items']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Avg field accuracy: {avg_field_accuracy:.2%}")
    print(f"Avg LLM score: {avg_llm_score:.1f}/10")
    print(f"\nResults available in Langfuse dashboard under:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Run: {run_name}")

    return summary


# =============================================================================
# UTILITY
# =============================================================================

def flush_langfuse():
    """Flush Langfuse events to ensure they're sent."""
    langfuse = get_client()
    if langfuse:
        langfuse.flush()
