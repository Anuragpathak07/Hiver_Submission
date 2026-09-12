import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline import SupportAgentPipeline
from escalation.escalation_engine import EscalationEngine
from retrieval.retrieval_engine import RetrievalEngine

def test_escalation_engine_sensitive_keyword():
    engine = EscalationEngine()
    result = engine.evaluate(
        customer_message="I demand a refund for my billing invoice",
        predicted_intent="battery_life_issue",
        classifier_confidence=0.95,
        retrieved_cases=[{"similarity_score": 0.85}]
    )
    assert result["escalate"] is True
    assert result["decision"] == "ESCALATE"
    assert result["rule_triggered"] == "SENSITIVE_KEYWORD"

def test_escalation_engine_high_risk_intent():
    engine = EscalationEngine()
    result = engine.evaluate(
        customer_message="Can you help me reset my account password?",
        predicted_intent="account_identity_issue",
        classifier_confidence=0.90,
        retrieved_cases=[{"similarity_score": 0.80}]
    )
    assert result["escalate"] is True
    assert result["decision"] == "ESCALATE"
    assert result["rule_triggered"] == "HIGH_RISK_INTENT"

def test_escalation_engine_auto_handle():
    engine = EscalationEngine()
    result = engine.evaluate(
        customer_message="My battery life is terrible after updating",
        predicted_intent="battery_life_issue",
        classifier_confidence=0.88,
        retrieved_cases=[{"similarity_score": 0.75}]
    )
    assert result["escalate"] is False
    assert result["decision"] == "AUTO_HANDLE"

def test_pipeline_integration():
    pipeline = SupportAgentPipeline(sample_retrieval_size=5000)
    query = "My battery drains very fast on my iPhone"
    output = pipeline.process_query(query)

    assert "customer_message" in output
    assert "intent_classification" in output
    assert "retrieval" in output
    assert "escalation" in output
    assert "response_generation" in output

    assert output["intent_classification"]["predicted_intent"] in [
        "battery_life_issue", "ios_update_issues", "iphone_hardware_issue", "other_unclear"
    ]
    assert len(output["retrieval"]["top_cases"]) > 0
