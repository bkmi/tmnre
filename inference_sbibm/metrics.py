import logging
from argparse import ArgumentParser
from pathlib import Path

import yaml

import sbibm
from tmnre.benchmark.metrics import compute_metrics_df
from tmnre.benchmark.paths import benchmark_paths, prepend_root


def main(root: Path, n_jobs: int):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    root = Path(root)
    rooted_benchmark_paths = prepend_root(root, benchmark_paths)

    with open(rooted_benchmark_paths.run_yaml, "r") as f:
        run_yaml = yaml.load(f, yaml.Loader)

    log.info(f'Task name {run_yaml["task"]["name"]}')
    log.info(f'Task parameters {run_yaml["task"]["params"]}')
    task = sbibm.get_task(run_yaml["task"]["name"], **run_yaml["task"]["params"])

    df_metrics = compute_metrics_df(
        task=task,
        num_observation=run_yaml["task"]["num_observation"],
        benchmark_paths=rooted_benchmark_paths,
        n_jobs=n_jobs,
        do_kld=False,
    )
    df_metrics.to_csv(rooted_benchmark_paths.metrics, index=False)
    log.info(f"Metrics:\n{df_metrics.transpose().to_string(header=False)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    main(args.directory, args.n_jobs)
