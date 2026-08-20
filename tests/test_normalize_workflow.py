import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.normalize_workflow import normalize_workflow, normalize_value

def test_workflow_name_preservation():
    pw = {"name": "Test Workflow"}
    text = normalize_workflow(pw)
    assert "[WORKFLOW_NAME]\nTest Workflow" in text

def test_trigger_preservation():
    pw = {"on": {"push": {"branches": ["main"]}}}
    text = normalize_workflow(pw)
    assert "[TRIGGERS]\npush: {branches: [main]}" in text

def test_permission_preservation():
    pw = {"permissions": {"contents": "read", "issues": "write"}}
    text = normalize_workflow(pw)
    assert "contents: read" in text
    assert "issues: write" in text

def test_action_and_command_preservation():
    pw = {
        "jobs": {
            "build": {
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {"run": "echo Hello"}
                ]
            }
        }
    }
    text = normalize_workflow(pw)
    assert "uses: actions/checkout@v4" in text
    assert "run: echo Hello" in text

def test_dependency_preservation():
    pw = {
        "jobs": {
            "test": {"needs": ["build"]}
        }
    }
    text = normalize_workflow(pw)
    assert "needs: [build]" in text

def test_step_ordering():
    pw = {
        "jobs": {
            "build": {
                "steps": [
                    {"name": "step1", "run": "cmd1"},
                    {"name": "step2", "run": "cmd2"}
                ]
            }
        }
    }
    text = normalize_workflow(pw)
    idx1 = text.find("cmd1")
    idx2 = text.find("cmd2")
    assert idx1 != -1 and idx2 != -1
    assert idx1 < idx2

def test_deterministic_ordering():
    pw = {
        "permissions": {"b": "2", "a": "1"}
    }
    text = normalize_workflow(pw)
    # Since dictionary ordering is deterministic (sorted), "a" should appear before "b"
    idx_a = text.find("a: 1")
    idx_b = text.find("b: 2")
    assert idx_a < idx_b

def test_deterministic_repeated():
    pw = {
        "name": "Testing",
        "jobs": {
            "a": {"name": "job a"},
            "b": {"name": "job b"}
        }
    }
    text1 = normalize_workflow(pw)
    text2 = normalize_workflow(pw)
    assert text1 == text2

def test_empty_optional_fields():
    pw = {}
    text = normalize_workflow(pw)
    assert "None" in text
    assert "[WORKFLOW_NAME]" in text

def test_normalize_value():
    assert normalize_value(True) == "True"
    assert normalize_value(False) == "False"
    assert normalize_value(" a  b \n c ") == "a b c"
    assert normalize_value({"z": 1, "a": 2}) == "{a: 2, z: 1}"
