#!/usr/bin/env python3
"""Report four-arm same-snapshot visual/original/zero/shuffle evaluation."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

def rate(n, d): return n / d if d else 0.0

def metrics(rows, arm):
    bs=[r["branches"][arm] for r in rows]
    n=len(bs); succ=sum(bool(x["success"]) for x in bs)
    aligned=sum(x["first_aligned_step"] is not None for x in bs)
    return {"n":n,"strict":succ,"strict_rate":rate(succ,n),"aligned":aligned,"alignment_rate":rate(aligned,n),
            "mean_success_steps": (sum(x["time_to_success_steps"] for x in bs if x["time_to_success_steps"])/sum(x["time_to_success_steps"] is not None for x in bs)) if any(x["time_to_success_steps"] is not None for x in bs) else None,
            "ejected":sum(bool(x["ejected"]) for x in bs),"passed_through":sum(bool(x["passed_through"]) for x in bs),"grasp_drift_failure":sum(bool(x["grasp_drift_failure"]) for x in bs)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--minimum-formal-pairs",type=int,default=64); ap.add_argument("--minimum-strict-delta",type=float,default=.10); a=ap.parse_args()
    src=json.loads(a.input.read_text()); modes=list(src.get("modes",["visual","torque_original","zero","shuffle"]))
    rows=[r for r in src["rows"] if r.get("paired_initial_observation_identical")]
    n=len(rows); integrity=n==src.get("pairs")==src.get("valid_identical_pairs")
    grid=Counter((int(r["sector"]),int(r["load_band"])) for r in rows); expected=n//16 if n and n%16==0 else None
    balanced=expected is not None and all(grid[(s,l)]==expected for s in range(8) for l in range(2))
    overall={m:metrics(rows,m) for m in modes}
    by_load={str(l):{m:metrics([r for r in rows if int(r["load_band"])==l],m) for m in modes} for l in range(2)}
    by_sector={str(s):{m:metrics([r for r in rows if int(r["sector"])==s],m) for m in modes} for s in range(8)}
    pairwise={}
    for m in modes:
        if m=="visual": continue
        w=sum(bool(r["branches"][m]["success"]) and not bool(r["branches"]["visual"]["success"]) for r in rows)
        loss=sum(bool(r["branches"]["visual"]["success"]) and not bool(r["branches"][m]["success"]) for r in rows)
        pairwise[m]={"wins_over_visual":w,"losses_to_visual":loss,"ties":n-w-loss}
    torque=overall.get("torque_original",{}); visual=overall.get("visual",{})
    delta=torque.get("strict_rate",0)-visual.get("strict_rate",0)
    both_loads=all(by_load[str(l)]["torque_original"]["strict_rate"]>by_load[str(l)]["visual"]["strict_rate"] for l in range(2)) if n else False
    sectors=sum(by_sector[str(s)]["torque_original"]["strict_rate"]>=by_sector[str(s)]["visual"]["strict_rate"] for s in range(8)) if n else 0
    safety=all(torque.get(k,0)<=visual.get(k,0) for k in ("ejected","passed_through","grasp_drift_failure"))
    formal=bool(n>=a.minimum_formal_pairs and integrity and balanced and delta>=a.minimum_strict_delta and pairwise.get("torque_original",{}).get("wins_over_visual",0)>pairwise.get("torque_original",{}).get("losses_to_visual",0) and both_loads and sectors>=6 and safety)
    trend=bool(integrity and balanced and delta>0 and pairwise.get("torque_original",{}).get("wins_over_visual",0)>pairwise.get("torque_original",{}).get("losses_to_visual",0) and safety)
    decision="PASS" if formal else "TREND" if trend else "FAIL"
    if n<a.minimum_formal_pairs: decision="SMOKE_PASS" if integrity and balanced else "SMOKE_FAIL"
    result={"decision":decision,"benchmark":src.get("benchmark"),"modes":modes,"pairs":n,"integrity":integrity,"balanced_sector_load_grid":balanced,"expected_per_cell":expected,"overall":overall,"by_load":by_load,"by_sector":by_sector,"pairwise_vs_visual":pairwise,"strict_rate_delta_torque_vs_visual":delta,"positive_both_loads":both_loads,"torque_noninferior_sectors":sectors,"safety_noninferior":safety,"thresholds":{"minimum_formal_pairs":a.minimum_formal_pairs,"minimum_strict_delta":a.minimum_strict_delta,"minimum_noninferior_sectors":6}}
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"decision.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Four-arm same-snapshot evaluation report","",f"- Decision: **{decision}**",f"- Valid identical pairs: {n}",f"- Grid balanced: {balanced}",""]
    for m in modes:
        q=overall[m]; lines.append(f"- {m}: {q['strict']}/{n} ({q['strict_rate']:.1%}), alignment {q['aligned']}/{n} ({q['alignment_rate']:.1%}), ejection/pass-through {q['ejected']}/{q['passed_through']}")
    lines += ["",f"- Torque minus visual strict-rate delta: {delta:+.1%}",f"- Torque paired wins/losses/ties vs visual: {pairwise.get('torque_original',{})}",f"- Torque positive in both load bands: {both_loads}",f"- Torque non-inferior sectors: {sectors}/8",f"- Torque safety non-inferior: {safety}","","Detailed load/sector and audit data are in `decision.json`."]
    (a.output_dir/"REPORT.md").write_text("\n".join(lines)+"\n"); print(json.dumps({"decision":decision,"pairs":n,"strict_delta":delta,"overall":overall}))
    if decision=="SMOKE_FAIL": raise SystemExit(2)
if __name__=="__main__": main()
