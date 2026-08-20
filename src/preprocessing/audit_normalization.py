import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

def deterministic_hash(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def canonical_json_hash(pw_json):
    # parse and re-dump with sort_keys=True
    pw = json.loads(pw_json)
    canonical = json.dumps(pw, sort_keys=True, separators=(',', ':'))
    return deterministic_hash(canonical)

def main():
    pq_path = Path("data/processed/labelled_workflow_versions_text_v1.parquet")
    print(f"Loading {pq_path}...")
    df = pd.read_parquet(pq_path)

    # 1. Hashes
    df["norm_hash"] = df["normalized_workflow_text"].apply(deterministic_hash)
    df["raw_hash"] = df["parsed_workflow_json"].apply(canonical_json_hash)

    norm_unique = df["norm_hash"].nunique()
    raw_unique = df["raw_hash"].nunique()

    # 2. Duplicate Groups
    group_sizes = df.groupby("norm_hash").size()
    duplicates = group_sizes[group_sizes > 1]
    
    print(f"Raw unique: {raw_unique}")
    print(f"Norm unique: {norm_unique}")
    print(f"Duplicate norm groups: {len(duplicates)}")

    cross_split_groups = 0
    affected_cross_split_records = 0
    affected_cross_split_repos = set()
    affected_cross_split_labels = set()

    group_records = []
    
    collisions = []

    for h, size in duplicates.items():
        sub = df[df["norm_hash"] == h]
        
        splits = sub["split"].unique()
        labels = sub["final_label"].unique()
        repos = sub["repository"].unique()
        w_ids = sub["workflow_id"].unique()
        sample_ids = sub["sample_id"].tolist()
        
        # Cross split check
        if len(splits) > 1:
            cross_split_groups += 1
            affected_cross_split_records += len(sub)
            affected_cross_split_repos.update(repos)
            affected_cross_split_labels.update(labels)
            
        # Collision check: same norm hash, different raw hash
        raw_hashes = sub["raw_hash"].unique()
        if len(raw_hashes) > 1:
            collisions.append({
                "norm_hash": h,
                "raw_hashes": list(raw_hashes),
                "size": size,
                "repos": list(repos),
                "w_ids": list(w_ids),
                "sample_ids": sample_ids
            })

        group_records.append({
            "normalized_text_hash": h,
            "group_size": size,
            "sample_ids": sample_ids,
            "repositories": list(repos),
            "workflow_ids": list(w_ids),
            "splits": list(splits),
            "labels": list(labels)
        })

    # Information Preservation
    source_triggers = 0
    norm_triggers = 0
    source_permissions = 0
    norm_permissions = 0
    source_actions = 0
    norm_actions = 0
    source_commands = 0
    norm_commands = 0
    source_jobs = 0
    norm_jobs = 0
    source_needs = 0
    norm_needs = 0
    
    # We will sample multiline commands to save to doc
    multiline_examples = []
    nested_examples = []
    
    for idx, row in df.iterrows():
        pw = json.loads(row["parsed_workflow_json"])
        text = row["normalized_workflow_text"]
        
        has_trigger = bool(pw.get("on") or pw.get("triggers"))
        has_perm = bool(pw.get("permissions"))
        
        if has_trigger: source_triggers += 1
        if has_perm: source_permissions += 1
        
        if "[TRIGGERS]\nNone" not in text and has_trigger:
            norm_triggers += 1
        if "[PERMISSIONS]\nNone" not in text and has_perm:
            norm_permissions += 1
            
        # Jobs
        jobs = pw.get("jobs", {})
        if jobs and isinstance(jobs, dict):
            source_jobs += 1
            if "JOB [" in text:
                norm_jobs += 1
                
            has_action = False
            has_cmd = False
            has_needs = False
            for jid, job in jobs.items():
                if isinstance(job, dict):
                    if job.get("needs"): has_needs = True
                    steps = job.get("steps")
                    if steps and isinstance(steps, list):
                        for s in steps:
                            if not isinstance(s, dict): continue
                            if s.get("uses"): has_action = True
                            if s.get("run"):
                                has_cmd = True
                                # check multiline
                                run_str = str(s.get("run"))
                                if "\n" in run_str and len(multiline_examples) < 20:
                                    multiline_examples.append({
                                        "source": run_str,
                                        "norm": [line for line in text.split("\n") if "run:" in line]
                                    })
                                    
            if has_needs: source_needs += 1
            if has_action: source_actions += 1
            if has_cmd: source_commands += 1
            
            if has_needs and "needs:" in text: norm_needs += 1
            if has_action and "uses:" in text: norm_actions += 1
            if has_cmd and "run:" in text: norm_commands += 1

            # nested examples
            if has_perm and len(nested_examples) < 5:
                nested_examples.append({
                    "type": "permissions",
                    "source": json.dumps(pw.get("permissions"), indent=2),
                    "norm": text.split("[PERMISSIONS]")[1].split("[")[0].strip()
                })
        
    print(f"\nCross split groups: {cross_split_groups}")
    print(f"Collisions: {len(collisions)}")
    
    # Save parquet
    df_dup = pd.DataFrame(group_records)
    out_pq = Path("data/intermediate/normalization_duplicate_groups_v1.parquet")
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    df_dup.to_parquet(out_pq, index=False)
    
    manifest = {
        "total_rows": len(df),
        "raw_unique": raw_unique,
        "norm_unique": norm_unique,
        "duplicate_norm_groups": len(duplicates),
        "largest_group": int(group_sizes.max()) if len(group_sizes) > 0 else 0,
        "cross_split_groups": cross_split_groups,
        "affected_cross_split_records": affected_cross_split_records,
        "collisions_raw_vs_norm": len(collisions),
        "information_preservation": {
            "triggers": {"source": source_triggers, "norm": norm_triggers},
            "permissions": {"source": source_permissions, "norm": norm_permissions},
            "jobs": {"source": source_jobs, "norm": norm_jobs},
            "actions": {"source": source_actions, "norm": norm_actions},
            "commands": {"source": source_commands, "norm": norm_commands},
            "needs": {"source": source_needs, "norm": norm_needs}
        },
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    out_man = Path("data/manifests/normalization_audit_v1.json")
    out_man.write_text(json.dumps(manifest, indent=4))
    
    # Markdown
    md = [
        "# Normalization Audit Report",
        "",
        "## Exact Duplicate Statistics",
        f"- Total rows: {len(df)}",
        f"- Unique Raw JSONs: {raw_unique}",
        f"- Unique Normalized Texts: {norm_unique}",
        f"- Duplicate Normalized Groups: {len(duplicates)}",
        f"- Largest Group Size: {group_sizes.max() if len(group_sizes) > 0 else 0}",
        "",
        "## Cross-Split Duplicate Statistics",
        f"- Groups crossing partitions: {cross_split_groups}",
        f"- Affected records: {affected_cross_split_records}",
        f"- Affected repositories: {len(affected_cross_split_repos)}",
        f"- Affected labels: {len(affected_cross_split_labels)}",
        "",
        "## Raw vs Normalized Collision Statistics",
        f"- Number of collisions (same norm text, different raw JSON): {len(collisions)}",
        ""
    ]
    
    if collisions:
        md.append("### Collision Sample (first 5)")
        for c in collisions[:5]:
            md.append(f"- Norm Hash: {c['norm_hash']}")
            md.append(f"  Raw Hashes: {c['raw_hashes']}")
            md.append(f"  Group Size: {c['size']}")
            md.append(f"  Repos: {c['repos']}")
            md.append(f"  Sample IDs: {c['sample_ids'][:3]}...")
            md.append("")
            
    md.extend([
        "## Information Preservation",
        f"- Triggers: Source={source_triggers}, Norm={norm_triggers}",
        f"- Permissions: Source={source_permissions}, Norm={norm_permissions}",
        f"- Jobs: Source={source_jobs}, Norm={norm_jobs}",
        f"- Actions: Source={source_actions}, Norm={norm_actions}",
        f"- Commands: Source={source_commands}, Norm={norm_commands}",
        f"- Needs: Source={source_needs}, Norm={norm_needs}",
        "",
        "## Nested Structure Findings"
    ])
    
    for ex in nested_examples:
        md.append(f"### Type: {ex['type']}")
        md.append("**SOURCE**")
        md.append("```json\n" + ex["source"] + "\n```")
        md.append("**NORMALIZED**")
        md.append("```\n" + ex["norm"] + "\n```")
        md.append("")
        
    md.append("## Command/Action Preservation (Multiline)")
    for i, ex in enumerate(multiline_examples[:5]):
        md.append(f"### Example {i+1}")
        md.append("**SOURCE COMMAND**")
        md.append("```\n" + ex["source"] + "\n```")
        md.append("**NORMALIZED LINES (found in text)**")
        for n in ex["norm"]:
            # truncate output if too long
            n_short = n if len(n) < 100 else n[:100] + "..."
            md.append(f"- `{n_short}`")
        md.append("")
        
    Path("docs/normalization_audit_v1.md").write_text("\n".join(md))
    print("Done!")

if __name__ == "__main__":
    main()
