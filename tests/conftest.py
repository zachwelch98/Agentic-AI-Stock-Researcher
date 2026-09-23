import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_langsmith_tracing_during_tests():
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # Local run logging stays off unless a test opts in via its own settings.
    os.environ.pop("LOCAL_TRACE_DIR", None)
