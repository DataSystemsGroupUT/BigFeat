"""
config.py
---------
Handles Ray initialization for Local, Remote Cluster, or Kubernetes.
"""

import os
import ray

def initialize_ray(options=None):
    """
    Initializes Ray based on the provided options or environment variables.
    Options can include:
    - address: 'auto' for local cluster, or a specific IP/service name for K8s.
    - num_cpus: number of CPUs to use.
    - ray_init_kwargs: dict of additional args for ray.init().
    """
    if ray.is_initialized():
        return

    options = options or {}
    # 1. Check for K8s or Remote Cluster via address
    # If in K8s, address is usually 'ray://ray-head-service:10001'
    # If Local, address is None
    address = options.get("address", os.environ.get("RAY_ADDRESS"))
    
    init_args = {
        "address": address,
        "ignore_reinit_error": True,
        "include_dashboard": False
    }
    
    # Update with any user-provided kwargs (like num_cpus, etc.)
    if "ray_init_kwargs" in options:
        init_args.update(options["ray_init_kwargs"])
        
    ray.init(**init_args)