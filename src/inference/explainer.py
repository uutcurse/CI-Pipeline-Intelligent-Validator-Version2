def generate_explanation(predicted_label: str, struct_features: dict) -> dict:
    """
    Generates a list of observable workflow signals to explain the predicted risk,
    based on exact feature values from the structural feature extractor.
    Does NOT alter predictions.
    """
    signals = []
    
    # Safely extract features with defaults
    jobs = struct_features.get('job_count', 0)
    steps = struct_features.get('step_count', 0)
    depth = struct_features.get('dependency_graph_depth', 0)
    matrix = struct_features.get('jobs_with_matrix', 0)
    parallel = struct_features.get('parallel_root_job_count', 0)
    permissions = struct_features.get('has_broad_write_permissions', 0)
    mutable = struct_features.get('actions_without_explicit_version', 0) + struct_features.get('actions_at_major_version', 0)
    
    if predicted_label == "LOW":
        if jobs > 0 and jobs <= 2:
            signals.append({"name": "Small workflow structure", "detail": f"Contains only {int(jobs)} job(s)"})
        if depth <= 1:
            signals.append({"name": "Limited dependency complexity", "detail": "Jobs have flat or minimal dependencies"})
        if matrix == 0:
            signals.append({"name": "Low matrix expansion", "detail": "No dynamic matrix execution detected"})
        signals = signals[:3]
        
    elif predicted_label == "MEDIUM":
        if jobs > 1 and jobs <= 5:
            signals.append({"name": "Multiple jobs", "detail": f"Workflow is split into {int(jobs)} parallel/sequential jobs"})
        if matrix > 0:
            signals.append({"name": "Matrix execution", "detail": "Matrix-based execution paths detected"})
        if depth > 1:
            signals.append({"name": "Moderate dependency depth", "detail": f"Dependency chain reaches depth of {int(depth)}"})
        if parallel > 1:
            signals.append({"name": "Multiple execution paths", "detail": "Concurrent jobs/steps present"})
        signals = signals[:4]
        
    elif predicted_label == "HIGH":
        if jobs > 5:
            signals.append({"name": "Large workflow", "detail": f"High complexity with {int(jobs)} jobs"})
        if steps > 20:
            signals.append({"name": "High step count", "detail": f"Total of {int(steps)} steps across all jobs"})
        if matrix > 0:
            signals.append({"name": "Matrix execution", "detail": "Expanded matrix strategy exponentially increases execution paths"})
        if depth > 2:
            signals.append({"name": "Deep job dependency chain", "detail": f"Complex dependency graph (depth {int(depth)})"})
        if permissions > 0:
            signals.append({"name": "Broad write permissions", "detail": "Elevated token permissions detected"})
        if mutable > 0:
            signals.append({"name": "Mutable third-party actions", "detail": "Floating tags or branches instead of fixed SHAs"})
        signals = signals[:5]
        
    return {
        "title": "Key Risk Signals Detected",
        "signals": signals
    }
