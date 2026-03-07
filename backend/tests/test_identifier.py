from exoqml.services.identifier import resolve_target


def test_resolve_tic_auto() -> None:
    target = resolve_target("TIC 12345")
    assert target.target_type == "tic"
    assert target.target_id == "12345"
    assert target.query == "TIC 12345"


def test_resolve_kic_auto() -> None:
    target = resolve_target("kic 87654")
    assert target.target_type == "kic"
    assert target.query == "KIC 87654"


def test_resolve_name_auto() -> None:
    target = resolve_target("Kepler-10")
    assert target.target_type == "name"
    assert target.query == "Kepler-10"
