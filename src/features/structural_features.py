import json
import re
import pandas as pd
import numpy as np

def _safe_len(obj):
    return len(obj) if obj else 0

def build_dependency_graph(jobs):
    nodes = list(jobs.keys()) if isinstance(jobs, dict) else []
    edges = []
    in_degree = {n: 0 for n in nodes}
    out_degree = {n: 0 for n in nodes}
    
    if isinstance(jobs, dict):
        for jid, job in jobs.items():
            if not isinstance(job, dict): continue
            needs = job.get("needs")
            if not needs: continue
            
            if isinstance(needs, str):
                needs = [needs]
            elif not isinstance(needs, list):
                continue
                
            for dep in needs:
                if isinstance(dep, str):
                    if dep not in in_degree:
                        in_degree[dep] = 0
                        out_degree[dep] = 0
                        nodes.append(dep)
                    edges.append((dep, jid))
                    out_degree[dep] += 1
                    in_degree[jid] += 1
                    
    return nodes, edges, in_degree, out_degree

def get_graph_depth(nodes, edges, in_degree):
    if not nodes: return 0
    # simple longest path in DAG
    dist = {n: 0 for n in nodes}
    # topological sort
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        
    in_d = in_degree.copy()
    queue = [n for n in nodes if in_d[n] == 0]
    
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            dist[v] = max(dist[v], dist[u] + 1)
            in_d[v] -= 1
            if in_d[v] == 0:
                queue.append(v)
                
    return max(dist.values()) if dist else 0

def extract_workflow_features(pw):
    feat = {}
    
    # 1. Size
    jobs = pw.get("jobs", {})
    if not isinstance(jobs, dict): jobs = {}
    
    j_count = len(jobs)
    step_counts = []
    for j in jobs.values():
        if isinstance(j, dict) and isinstance(j.get("steps"), list):
            step_counts.append(len(j.get("steps")))
        else:
            step_counts.append(0)
            
    s_count = sum(step_counts)
    
    feat["job_count"] = j_count
    feat["step_count"] = s_count
    feat["average_steps_per_job"] = (s_count / j_count) if j_count > 0 else 0.0
    feat["max_steps_per_job"] = max(step_counts) if step_counts else 0
    
    name = pw.get("name")
    feat["workflow_has_name"] = 1 if name else 0
    feat["workflow_name_length"] = len(str(name)) if name else 0
    
    # 2. Triggers
    on_val = pw.get("on")
    triggers_array = pw.get("triggers", [])
    
    trig_set = set()
    if isinstance(on_val, dict):
        trig_set.update(on_val.keys())
    elif isinstance(on_val, list):
        trig_set.update(str(x) for x in on_val)
    elif isinstance(on_val, str):
        trig_set.add(on_val)
        
    if isinstance(triggers_array, list):
        trig_set.update(str(x) for x in triggers_array)
        
    feat["trigger_count"] = len(trig_set)
    feat["trigger_push"] = 1 if "push" in trig_set else 0
    feat["trigger_pull_request"] = 1 if "pull_request" in trig_set else 0
    feat["trigger_pull_request_target"] = 1 if "pull_request_target" in trig_set else 0
    feat["trigger_workflow_dispatch"] = 1 if "workflow_dispatch" in trig_set else 0
    feat["trigger_schedule"] = 1 if "schedule" in trig_set else 0
    feat["trigger_workflow_call"] = 1 if "workflow_call" in trig_set else 0
    feat["trigger_repository_dispatch"] = 1 if "repository_dispatch" in trig_set else 0
    feat["has_multiple_triggers"] = 1 if len(trig_set) > 1 else 0
    
    # 3. Dependencies
    nodes, edges, in_degree, out_degree = build_dependency_graph(jobs)
    feat["dependency_edge_count"] = len(edges)
    feat["jobs_with_dependencies"] = sum(1 for d in in_degree.values() if d > 0)
    feat["jobs_without_dependencies"] = sum(1 for d in in_degree.values() if d == 0)
    feat["max_dependency_fan_in"] = max(in_degree.values()) if in_degree else 0
    feat["max_dependency_fan_out"] = max(out_degree.values()) if out_degree else 0
    feat["dependency_graph_depth"] = get_graph_depth(nodes, edges, in_degree)
    feat["has_dependency_graph"] = 1 if len(edges) > 0 else 0
    feat["parallel_root_job_count"] = sum(1 for n in nodes if in_degree.get(n) == 0 and out_degree.get(n) > 0)
    feat["sink_job_count"] = sum(1 for n in nodes if in_degree.get(n) > 0 and out_degree.get(n) == 0)
    
    # 4. Runners
    runners = set()
    self_hosted = 0
    gh_hosted = 0
    
    for j in jobs.values():
        if isinstance(j, dict):
            ro = j.get("runs_on")
            if isinstance(ro, str):
                runners.add(ro)
                if "self-hosted" in ro.lower(): self_hosted += 1
                else: gh_hosted += 1
            elif isinstance(ro, list):
                runners.update(str(x) for x in ro)
                if any("self-hosted" in str(x).lower() for x in ro): self_hosted += 1
                else: gh_hosted += 1
                
    feat["unique_runner_count"] = len(runners)
    feat["self_hosted_runner_count"] = self_hosted
    feat["github_hosted_runner_count"] = gh_hosted
    feat["runner_diversity"] = len(runners)
    
    # 5 & 6. Actions and Commands
    total_actions = 0
    unique_actions = set()
    first_party = 0
    third_party = 0
    local_action = 0
    docker_action = 0
    without_version = 0
    major_version = 0
    sha_version = 0
    
    run_steps = 0
    shells = set()
    multiline = 0
    cmd_lens = []
    cmd_tokens = 0
    
    # Caching and artifacts
    cache_count = 0
    art_up = 0
    art_down = 0
    retry_count = 0
    
    for j in jobs.values():
        if not isinstance(j, dict): continue
        steps = j.get("steps")
        if not isinstance(steps, list): continue
        
        for s in steps:
            if not isinstance(s, dict): continue
            
            uses = s.get("uses")
            if uses and isinstance(uses, str):
                total_actions += 1
                unique_actions.add(uses)
                
                if uses.startswith("actions/"): first_party += 1
                elif uses.startswith("./") or uses.startswith(".github/"): local_action += 1
                elif uses.startswith("docker://"): docker_action += 1
                else: third_party += 1
                
                if "@" not in uses and not uses.startswith("./") and not uses.startswith("docker://"):
                    without_version += 1
                elif "@v" in uses and len(uses.split("@v")[-1]) <= 2:
                    major_version += 1
                elif "@" in uses and len(uses.split("@")[-1]) == 40: # sha
                    sha_version += 1
                    
                if "actions/cache" in uses: cache_count += 1
                if "actions/upload-artifact" in uses: art_up += 1
                if "actions/download-artifact" in uses: art_down += 1
                if "retry" in uses.lower(): retry_count += 1
                
            run = s.get("run")
            if run and isinstance(run, str):
                run_steps += 1
                if "\n" in run.strip(): multiline += 1
                cmd_lens.append(len(run))
                cmd_tokens += len(run.split())
                
                if "retry" in run.lower(): retry_count += 1
                
            shell = s.get("shell")
            if shell and isinstance(shell, str):
                shells.add(shell)
                
    feat["total_action_steps"] = total_actions
    feat["unique_action_count"] = len(unique_actions)
    feat["first_party_action_count"] = first_party
    feat["third_party_action_count"] = third_party
    feat["local_action_count"] = local_action
    feat["docker_action_count"] = docker_action
    feat["action_version_count"] = total_actions - without_version - local_action - docker_action
    feat["actions_without_explicit_version"] = without_version
    feat["actions_at_major_version"] = major_version
    feat["actions_at_sha"] = sha_version
    
    feat["run_step_count"] = run_steps
    feat["unique_shell_count"] = len(shells)
    feat["multiline_command_count"] = multiline
    feat["average_command_length"] = np.mean(cmd_lens) if cmd_lens else 0.0
    feat["max_command_length"] = max(cmd_lens) if cmd_lens else 0
    feat["command_token_count"] = cmd_tokens
    
    feat["cache_action_count"] = cache_count
    feat["artifact_upload_count"] = art_up
    feat["artifact_download_count"] = art_down
    
    # 7. Conditions
    cond_j = 0
    cond_s = 0
    for j in jobs.values():
        if not isinstance(j, dict): continue
        if j.get("if"): cond_j += 1
        steps = j.get("steps")
        if isinstance(steps, list):
            for s in steps:
                if isinstance(s, dict) and s.get("if"):
                    cond_s += 1
                    
    feat["conditional_job_count"] = cond_j
    feat["conditional_step_count"] = cond_s
    feat["total_condition_count"] = cond_j + cond_s
    total_elements = j_count + s_count
    feat["conditional_branch_density"] = (cond_j + cond_s) / total_elements if total_elements > 0 else 0.0
    
    # 8. Reliability
    timeout_c = 0
    cont_err_j = 0
    cont_err_s = 0
    
    for j in jobs.values():
        if not isinstance(j, dict): continue
        if j.get("timeout_minutes") is not None: timeout_c += 1
        if j.get("continue_on_error") is not None: cont_err_j += 1
        
        steps = j.get("steps")
        if isinstance(steps, list):
            for s in steps:
                if isinstance(s, dict):
                    if s.get("timeout_minutes") is not None: timeout_c += 1
                    if s.get("continue_on_error") is not None: cont_err_s += 1
                    
    feat["timeout_configured_count"] = timeout_c
    feat["job_continue_on_error_count"] = cont_err_j
    feat["step_continue_on_error_count"] = cont_err_s
    feat["continue_on_error_count"] = cont_err_j + cont_err_s
    feat["retry_related_configuration_count"] = retry_count
    
    # 9. Security
    perms = pw.get("permissions")
    feat["permissions_present"] = 1 if perms is not None else 0
    
    reads = 0
    writes = 0
    nones = 0
    has_write = 0
    broad_write = 0
    
    if isinstance(perms, str):
        if perms == "read-all": reads = 1
        elif perms == "write-all": 
            writes = 1
            has_write = 1
            broad_write = 1
    elif isinstance(perms, dict):
        for v in perms.values():
            if v == "read": reads += 1
            elif v == "write": 
                writes += 1
                has_write = 1
            elif v == "none": nones += 1
            
    feat["permissions_scope_count"] = len(perms) if isinstance(perms, dict) else (1 if perms else 0)
    feat["permissions_read_count"] = reads
    feat["permissions_write_count"] = writes
    feat["permissions_none_count"] = nones
    feat["has_write_permission"] = has_write
    feat["has_broad_write_permissions"] = broad_write
    
    # 10. Environment
    g_env = pw.get("env")
    feat["global_env_variable_count"] = len(g_env) if isinstance(g_env, dict) else 0
    
    j_env_c = 0
    s_env_c = 0
    env_names = set()
    
    # Secret references
    # naive regex search for secrets
    pw_str = json.dumps(pw)
    feat["secret_reference_count"] = len(re.findall(r"\$\{\{\s*secrets\.", pw_str))
    
    for j in jobs.values():
        if not isinstance(j, dict): continue
        if j.get("env"): j_env_c += 1
        
        env = j.get("environment")
        if isinstance(env, str): env_names.add(env)
        elif isinstance(env, dict) and "name" in env: env_names.add(env["name"])
        
        steps = j.get("steps")
        if isinstance(steps, list):
            for s in steps:
                if isinstance(s, dict) and s.get("env"): s_env_c += 1
                
    feat["jobs_with_env_count"] = j_env_c
    feat["steps_with_env_count"] = s_env_c
    feat["environment_name_count"] = len(env_names)
    
    # 11. Services / Containers
    j_cont = 0
    j_serv = 0
    serv_c = 0
    u_serv = set()
    
    for j in jobs.values():
        if not isinstance(j, dict): continue
        if j.get("container"): j_cont += 1
        
        srv = j.get("services")
        if isinstance(srv, dict):
            j_serv += 1
            serv_c += len(srv)
            for sname, sdef in srv.items():
                if isinstance(sdef, dict) and sdef.get("image"):
                    u_serv.add(sdef["image"])
                    
    feat["job_container_count"] = j_cont
    feat["jobs_with_services"] = j_serv
    feat["service_count"] = serv_c
    feat["unique_service_count"] = len(u_serv)
    
    # 12. Matrix
    mat_j = 0
    strat_j = 0
    mat_dim = 0
    mat_comb = 0
    ff = 0
    mp = 0
    
    for j in jobs.values():
        if not isinstance(j, dict): continue
        strat = j.get("strategy")
        if isinstance(strat, dict):
            strat_j += 1
            if strat.get("fail-fast") is not None: ff += 1
            if strat.get("max-parallel") is not None: mp += 1
            
            mat = strat.get("matrix")
            if isinstance(mat, dict):
                mat_j += 1
                mat_dim += len(mat)
                
                # estimate combinations
                comb = 1
                known = True
                for vals in mat.values():
                    if isinstance(vals, list):
                        comb *= len(vals)
                    else:
                        known = False
                
                inc = strat.get("include")
                if isinstance(inc, list):
                    comb += len(inc)
                
                if known: mat_comb += comb
                
    feat["jobs_with_strategy"] = strat_j
    feat["jobs_with_matrix"] = mat_j
    feat["matrix_dimension_count"] = mat_dim
    feat["estimated_matrix_combinations"] = mat_comb
    feat["fail_fast_configured"] = ff
    feat["max_parallel_configured"] = mp
    
    # 13. Concurrency
    w_conc = pw.get("concurrency")
    feat["workflow_concurrency_present"] = 1 if w_conc is not None else 0
    j_conc = 0
    for j in jobs.values():
        if isinstance(j, dict) and j.get("concurrency") is not None:
            j_conc += 1
    feat["job_concurrency_count"] = j_conc
    feat["concurrency_configured"] = 1 if w_conc is not None or j_conc > 0 else 0
    
    # 15. Language
    langs = pw.get("languages", {})
    feat["language_count"] = len(langs) if isinstance(langs, dict) else 0
    feat["language_diversity"] = len(langs) if isinstance(langs, dict) else 0
    feat["declared_language_presence"] = 1 if feat["language_count"] > 0 else 0

    # Safety cast for parquet serialization
    for k, v in feat.items():
        if isinstance(v, (int, float, bool)):
            feat[k] = float(v)
            
    return feat
