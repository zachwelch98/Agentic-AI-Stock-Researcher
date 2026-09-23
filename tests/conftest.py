import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_langsmith_tracing_during_tests():
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "false"
