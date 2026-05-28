from pathlib import Path

from src.app.ignite import load_utahfile, verify_hooks


def test_utahfile_v6_hooks():
    manifest = load_utahfile(Path("Utahfile"))
    assert manifest["schema_version"] == "v6.Omnibus_Adelic"
    audit = verify_hooks(manifest)
    assert all(audit.values()), audit
