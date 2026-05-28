from src.core.capital_sieve import AssetPosition, AutonomousAuditor


def test_audit_emits_intent_when_drag_exceeds_threshold():
    risk = {"a": 0.0, "b": 0.0}
    auditor = AutonomousAuditor(efficiency_threshold=100.0, venue_risk_matrix=risk)
    positions = [
        AssetPosition("USD", "a", 1_000_000.0, 0, 0.04),
        AssetPosition("USD", "b", 1_000_000.0, 0, 0.06),
    ]
    intents = auditor.audit_portfolio_drag(positions)
    assert len(intents) == 1
    assert intents[0].source_venue == "a"
    assert intents[0].destination_venue == "b"


def test_single_venue_no_intent():
    auditor = AutonomousAuditor(100.0, {})
    positions = [AssetPosition("USD", "a", 1.0, 0, 0.05)]
    assert auditor.audit_portfolio_drag(positions) == []
