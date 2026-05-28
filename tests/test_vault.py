from src.app.vault_daemon import SovereignVault


def test_threshold_signing():
    vault = SovereignVault(required_signatures=2)
    vault.broadcast_intent("x", {"amount": 1})
    vault.sign("x", "a")
    assert len(vault.ready_to_execute()) == 0
    vault.sign("x", "b")
    assert len(vault.ready_to_execute()) == 1
