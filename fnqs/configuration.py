"""Validate experiment configs before training, allocation or artifact writes."""

from __future__ import annotations

import numpy as np

from .basis import validate_size
from .hamiltonians import IsingCouplings
from .models import GRUConfig


def validate_config(command: str, config: dict) -> None:
    """Reject invalid/unknown keys and train/test leakage at CLI boundaries."""
    common={"sizes"}
    required={
        "reference":common|{"window_x","points"},
        "train":common|{"model","train_fields","test_fields","seeds","steps","learning_rate","mode","optimizer","samples_per_task","audit_samples"},
        "recon":common|{"fields","families"},
    }
    optional={"reference":set(),"train":{"test_sizes","collapse_window_x","sr_shift",
              "sr_solver","sr_tolerance","sr_maxiter","sr_max_step_norm","sr_backtracking",
              "tasks_per_step","warm_start_checkpoint"},"recon":set()}
    if command not in required or not isinstance(config,dict):
        raise ValueError("invalid experiment command or config")
    missing=required[command]-config.keys();extra=config.keys()-required[command]-optional[command]
    if missing or extra:
        raise ValueError(f"missing keys={sorted(missing)}, unknown keys={sorted(extra)}")
    sizes=config["sizes"]
    if not isinstance(sizes,list) or not sizes or len(set(sizes))!=len(sizes):
        raise ValueError("sizes must be a nonempty unique list")
    for n in sizes:validate_size(n)
    if command=="reference":
        if len(sizes)<4 or min(sizes)<16 or config["points"]<21 or not isinstance(config["points"],int) or not 0<config["window_x"]<min(sizes):
            raise ValueError("reference requires >=4 sizes>=16, >=21 grid points and a positive window")
    elif command=="train":
        GRUConfig(**config["model"])
        checkpoint=config.get("warm_start_checkpoint")
        if checkpoint is not None and (not isinstance(checkpoint,str) or not checkpoint.strip()):
            raise ValueError("warm_start_checkpoint must be a nonempty path")
        if config["mode"] not in {"exact","vmc"} or config["optimizer"] not in {"adam","sr"}:
            raise ValueError("mode must be exact/vmc, optimizer adam/sr")
        for name in ["steps","samples_per_task","audit_samples"]:
            if isinstance(config[name],bool) or not isinstance(config[name],int) or config[name]< (1 if name=="steps" else 2):
                raise ValueError(f"{name} must be a positive integer (samples>=2)")
        for name in ["learning_rate","sr_shift","collapse_window_x"]:
            value=config.get(name,1e-3 if name=="sr_shift" else 1)
            if not np.isfinite(value) or value<=0:raise ValueError(f"{name} must be finite and positive")
        if config.get("sr_solver","dense") not in {"dense","cg"}:
            raise ValueError("sr_solver must be dense/cg")
        tolerance=config.get("sr_tolerance",1e-8)
        if not np.isfinite(tolerance) or not 0<tolerance<1:
            raise ValueError("sr_tolerance must be between zero and one")
        maxiter=config.get("sr_maxiter",500)
        if isinstance(maxiter,bool) or not isinstance(maxiter,int) or maxiter<1:
            raise ValueError("sr_maxiter must be a positive integer")
        limit=config.get("sr_max_step_norm")
        if limit is not None and (not np.isfinite(limit) or limit<=0):
            raise ValueError("sr_max_step_norm must be finite and positive")
        backtracking=config.get("sr_backtracking",False)
        if not isinstance(backtracking,bool) or (backtracking and (config["mode"]!="exact" or config["optimizer"]!="sr")):
            raise ValueError("sr_backtracking requires exact SR")
        train_fields,test_fields=config["train_fields"],config["test_fields"]
        if not train_fields or not test_fields or set(train_fields)&set(test_fields):
            raise ValueError("nonempty disjoint train/test fields required")
        for fields in [train_fields,test_fields]:
            if len(set(fields))!=len(fields) or not np.all(np.isfinite(fields)) or min(fields)<=0:
                raise ValueError("fields must be unique finite and positive")
        batch=config.get("tasks_per_step")
        if batch is not None and (isinstance(batch,bool) or not isinstance(batch,int) or
                not 1<=batch<=len(sizes)*len(train_fields)):
            raise ValueError("tasks_per_step must fit the training ensemble")
        test_sizes=config.get("test_sizes",[])
        if set(test_sizes)&set(sizes) or len(set(test_sizes))!=len(test_sizes):
            raise ValueError("held sizes must be unique and absent from training")
        for n in test_sizes:validate_size(n)
        if config["mode"]=="exact" and max(sizes+test_sizes)>10:
            raise ValueError("exact mode requires sizes<=10")
        seeds=config["seeds"]
        if not seeds or len(set(seeds))!=len(seeds) or any(isinstance(s,bool) or not isinstance(s,int) or not 0<=s<2**32 for s in seeds):
            raise ValueError("seeds must be unique nonnegative uint32 integers")
    else:
        if max(sizes)>14:raise ValueError("recon ED requires sizes<=14")
        fields=np.asarray(config["fields"],dtype=float)
        if fields.ndim!=1 or len(fields)<5 or not np.all(np.isfinite(fields)) or np.any(np.diff(fields)<=0) or min(fields)<=0:
            raise ValueError("need >=5 positive strictly increasing finite fields")
        families=config["families"]
        if not families or len({f["name"] for f in families})!=len(families):
            raise ValueError("family names must be unique")
        for family in families:
            if set(family)-{"name","j2","jx"}:raise ValueError("unknown family keys")
            IsingCouplings(j2=family.get("j2",0),jx=family.get("jx",0))
