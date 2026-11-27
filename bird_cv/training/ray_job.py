"""Ray for one SLURM job."""

import os

import ray


def init_job_ray() -> None:
    """Start a private Ray instance sized to this SLURM job, if none is running.

    Ray otherwise sizes itself to the whole node (all its cores), starting
    far more worker processes than the job's CPUs, and a fixed-port
    ``ray start --head`` collides with other jobs on the node. A local
    instance on free ports, with the job's CPU count and a small object
    store, avoids both; it shuts down with the process.
    """
    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
            include_dashboard=False,
            # Tuning passes almost no data through Ray; a small object store
            # keeps two jobs on one node within shared memory.
            object_store_memory=2 * 1024**3,
        )
