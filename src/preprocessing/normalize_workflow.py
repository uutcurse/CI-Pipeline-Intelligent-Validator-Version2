import json
import pandas as pd
from pathlib import Path

def normalize_value(val):
    if val is None:
        return "None"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        val = " ".join(val.split())
        return val
    if isinstance(val, list):
        return "[" + ", ".join([normalize_value(v) for v in val]) + "]"
    if isinstance(val, dict):
        items = [f"{k}: {normalize_value(v)}" for k, v in sorted(val.items())]
        return "{" + ", ".join(items) + "}"
    return str(val)

def normalize_dict_block(d):
    if not d:
        return "None"
    if isinstance(d, dict):
        items = [f"{k}: {normalize_value(v)}" for k, v in sorted(d.items())]
        return "\n".join(items)
    if isinstance(d, list):
        return "\n".join(normalize_value(v) for v in d)
    return str(d)

def normalize_triggers(pw):
    on_val = pw.get("on")
    triggers = pw.get("triggers")
    
    if isinstance(on_val, dict) and len(on_val) > 0:
        return normalize_dict_block(on_val)
    elif isinstance(on_val, list) and len(on_val) > 0:
        return "\n".join(sorted([str(t) for t in on_val]))
    elif isinstance(on_val, str):
        return on_val
    elif isinstance(triggers, list) and len(triggers) > 0:
        return "\n".join(sorted([str(t) for t in triggers]))
    return "None"

def normalize_jobs(jobs_dict):
    if not jobs_dict: return "None"
    if not isinstance(jobs_dict, dict): return str(jobs_dict)
    
    lines = []
    for job_id, job in sorted(jobs_dict.items()):
        lines.append(f"JOB [{job_id}]")
        
        # job might be a string in malformed yamls that bypass parsers
        if not isinstance(job, dict):
            lines.append(f"  malformed_job: {normalize_value(job)}\n")
            continue
            
        name = job.get("name")
        if name: lines.append(f"  name: {normalize_value(name)}")
            
        runs_on = job.get("runs_on")
        if runs_on: lines.append(f"  runs_on: {normalize_value(runs_on)}")
            
        needs = job.get("needs")
        if needs: lines.append(f"  needs: {normalize_value(needs)}")
            
        if_cond = job.get("if")
        if if_cond: lines.append(f"  condition: {normalize_value(if_cond)}")
            
        env_val = job.get("environment")
        if env_val: lines.append(f"  environment: {normalize_value(env_val)}")
            
        concurrency = job.get("concurrency")
        if concurrency: lines.append(f"  concurrency: {normalize_value(concurrency)}")
            
        timeout = job.get("timeout_minutes")
        if timeout: lines.append(f"  timeout_minutes: {normalize_value(timeout)}")
            
        strategy = job.get("strategy")
        if strategy: lines.append(f"  strategy: {normalize_value(strategy)}")
            
        cont_error = job.get("continue_on_error")
        if cont_error is not None: lines.append(f"  continue_on_error: {normalize_value(cont_error)}")
            
        container = job.get("container")
        if container: lines.append(f"  container: {normalize_value(container)}")
            
        services = job.get("services")
        if services: lines.append(f"  services: {normalize_value(services)}")
            
        job_env = job.get("env")
        if job_env: lines.append(f"  env: {normalize_value(job_env)}")
            
        job_defaults = job.get("defaults")
        if job_defaults: lines.append(f"  defaults: {normalize_value(job_defaults)}")
            
        steps = job.get("steps")
        if steps and isinstance(steps, list):
            lines.append("  STEPS:")
            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_lines = []
                s_name = step.get("name")
                if s_name: step_lines.append(f"name: {normalize_value(s_name)}")
                
                s_id = step.get("id")
                if s_id: step_lines.append(f"id: {normalize_value(s_id)}")
                    
                s_if = step.get("if")
                if s_if: step_lines.append(f"condition: {normalize_value(s_if)}")
                    
                s_uses = step.get("uses")
                if s_uses: step_lines.append(f"uses: {normalize_value(s_uses)}")
                    
                s_run = step.get("run")
                if s_run: step_lines.append(f"run: {normalize_value(s_run)}")
                    
                s_shell = step.get("shell")
                if s_shell: step_lines.append(f"shell: {normalize_value(s_shell)}")
                    
                s_with = step.get("with")
                if s_with: step_lines.append(f"with: {normalize_value(s_with)}")
                    
                s_env = step.get("env")
                if s_env: step_lines.append(f"env: {normalize_value(s_env)}")
                    
                s_timeout = step.get("timeout_minutes")
                if s_timeout: step_lines.append(f"timeout_minutes: {normalize_value(s_timeout)}")
                    
                s_cont_error = step.get("continue_on_error")
                if s_cont_error is not None: step_lines.append(f"continue_on_error: {normalize_value(s_cont_error)}")
                    
                s_workdir = step.get("working_directory")
                if s_workdir: step_lines.append(f"working_directory: {normalize_value(s_workdir)}")
                
                if step_lines:
                    lines.append("    - " + " | ".join(step_lines))
        
        lines.append("")
        
    return "\n".join(lines).strip()

def normalize_workflow(pw):
    sections = []
    
    name = pw.get("name")
    sections.append("[WORKFLOW_NAME]")
    sections.append(normalize_value(name) if name else "None")
    
    sections.append("\n[TRIGGERS]")
    sections.append(normalize_triggers(pw))
    
    sections.append("\n[PERMISSIONS]")
    sections.append(normalize_dict_block(pw.get("permissions")))
    
    sections.append("\n[GLOBAL_ENV]")
    sections.append(normalize_dict_block(pw.get("env")))
    
    sections.append("\n[DEFAULTS]")
    sections.append(normalize_dict_block(pw.get("defaults")))
    
    sections.append("\n[CONCURRENCY]")
    conc = pw.get("concurrency")
    sections.append(normalize_value(conc) if conc else "None")
    
    sections.append("\n[WORKFLOW_METADATA]")
    meta = {}
    if "job_count" in pw: meta["job_count"] = pw["job_count"]
    if "step_count" in pw: meta["step_count"] = pw["step_count"]
    if pw.get("languages"): meta["languages"] = pw["languages"]
    if pw.get("services"): meta["services"] = pw["services"]
    if pw.get("runners"): meta["runners"] = pw["runners"]
    if pw.get("actions_used"): meta["actions_used"] = pw["actions_used"]
    sections.append(normalize_dict_block(meta))
    
    sections.append("\n[JOBS]")
    sections.append(normalize_jobs(pw.get("jobs", {})))
    
    return "\n".join(sections)
