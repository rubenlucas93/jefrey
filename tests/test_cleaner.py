import pytest
from cleaner.processor import Cleaner

def test_cleaner_fixes_exceptico():
    c = Cleaner(brain_model='llama3.2:latest', language='es')
    test_input = "[Speaker 1] 0.0-5.0: yo soy muy exceptico"
    output = c.process(test_input)
    assert "escéptico" in output.lower(), f"Expected 'escéptico' in {output}"
    assert "[Speaker 1]" in output, f"Expected '[Speaker 1]' tag in {output}"

def test_cleaner_keeps_tags():
    c = Cleaner(brain_model='llama3.2:latest', language='es')
    test_input = "[Speaker 2] 1.0-4.0: ehh hola que tal"
    output = c.process(test_input)
    assert "[Speaker 2]" in output, f"Lost speaker tag in: {output}"
    assert "hola" in output.lower()
