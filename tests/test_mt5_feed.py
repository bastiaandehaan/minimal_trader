import pytest

@pytest.mark.skipif(__import__("importlib").util.find_spec("MetaTrader5") is None,
                    reason="MetaTrader5 not installed")


def test_mt5_loader_imports():
    import feeds.mt5_feed as m
    assert hasattr(m, "fetch")  # Change to 'fetch'
