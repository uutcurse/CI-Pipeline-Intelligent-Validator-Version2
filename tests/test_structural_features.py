import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.structural_features import extract_workflow_features, build_dependency_graph

def test_simple_single_job():
    pw = {"jobs": {"build": {"steps": [{"run": "echo hello"}]}}}
    f = extract_workflow_features(pw)
    assert f["job_count"] == 1
    assert f["step_count"] == 1
    assert f["run_step_count"] == 1

def test_dependency_graph():
    jobs = {
        "a": {},
        "b": {"needs": "a"},
        "c": {"needs": ["a", "b"]},
        "d": {"needs": "c"}
    }
    nodes, edges, ind, outd = build_dependency_graph(jobs)
    assert len(nodes) == 4
    assert ("a", "b") in edges
    assert ("b", "c") in edges
    assert ("c", "d") in edges
    
    pw = {"jobs": jobs}
    f = extract_workflow_features(pw)
    assert f["dependency_edge_count"] == 4
    assert f["has_dependency_graph"] == 1
    assert f["dependency_graph_depth"] == 3 # a->b->c->d

def test_matrix_strategy():
    pw = {
        "jobs": {
            "test": {
                "strategy": {
                    "matrix": {
                        "os": ["ubuntu", "windows"],
                        "node": [12, 14, 16]
                    }
                }
            }
        }
    }
    f = extract_workflow_features(pw)
    assert f["jobs_with_strategy"] == 1
    assert f["jobs_with_matrix"] == 1
    assert f["matrix_dimension_count"] == 2
    assert f["estimated_matrix_combinations"] == 6

def test_permissions():
    pw = {
        "permissions": {
            "contents": "read",
            "issues": "write",
            "pages": "none"
        }
    }
    f = extract_workflow_features(pw)
    assert f["permissions_present"] == 1
    assert f["permissions_scope_count"] == 3
    assert f["permissions_read_count"] == 1
    assert f["permissions_write_count"] == 1
    assert f["permissions_none_count"] == 1
    assert f["has_write_permission"] == 1
    assert f["has_broad_write_permissions"] == 0

def test_leakage_absence():
    pw = {
        "jobs": {"a": {}},
        "final_label": 0,
        "failure_rate": 0.5
    }
    f = extract_workflow_features(pw)
    assert "failure_rate" not in f
    assert "final_label" not in f

def test_language_extraction():
    pw = {"languages": {"python": True, "javascript": True}}
    f = extract_workflow_features(pw)
    assert f["language_count"] == 2
    assert f["language_diversity"] == 2
    assert f["declared_language_presence"] == 1

def test_missing_vs_zero_behavior():
    pw = {}
    f = extract_workflow_features(pw)
    assert f["job_count"] == 0
    assert f["permissions_present"] == 0
    # No NaNs or crashes
    assert isinstance(f["job_count"], float) or isinstance(f["job_count"], int)

def test_heuristic_classification():
    pw = {
        "jobs": {
            "test": {
                "steps": [
                    {"uses": "actions/checkout@v4"}, # first party major
                    {"uses": "docker://alpine:latest"}, # docker
                    {"uses": "./.github/actions/setup"}, # local
                    {"uses": "some/third-party@abcdef1234567890abcdef1234567890abcdef12"} # third party sha
                ]
            }
        }
    }
    f = extract_workflow_features(pw)
    assert f["first_party_action_count"] == 1
    assert f["docker_action_count"] == 1
    assert f["local_action_count"] == 1
    assert f["third_party_action_count"] == 1
    assert f["actions_at_major_version"] == 1
    assert f["actions_at_sha"] == 1
