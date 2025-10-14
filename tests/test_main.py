import pytest
from checkpointing import main

@pytest.mark.parametrize(["inp", "expected"], [(0, 1), (1, 2), (5, 6)])
def test_main(inp:int, expected:int):
    assert main(inp) == expected