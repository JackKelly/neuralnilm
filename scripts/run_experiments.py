#!/usr/bin/env python
from __future__ import print_function, division
from os.path import join
import sys
from neuralnilm.config import config

import logging
logger = logging.getLogger('neuralnilm')

experiment_definition_path = config.get('Paths', 'experiment_definitions')
job_list_filename = join(experiment_definition_path, 'job_list.txt')


def main():
    if experiment_definition_path not in sys.path:
        sys.path.insert(0, experiment_definition_path)

    next_job = _get_next_job()
    while next_job:
        try:
            _run_job(next_job)
        except KeyboardInterrupt:
            delete_this_job = raw_input(
                "Delete this job from job list [Y/n]? ").lower()
            if delete_this_job != "n":
                _delete_completed_job()
            continue_running = raw_input(
                "Continue running other experiments [N/y]? ").lower()
            if continue_running != "y":
                break
        else:
            _delete_completed_job()
            next_job = _get_next_job()


def _get_next_job():
    with open(job_list_filename, 'r') as fh:
        next_job = fh.readline().strip()
    return next_job


def _run_job(next_job):
    logger.info("Running {}".format(next_job))
    exec("import {:s}".format(next_job))
    from neuralnilm.utils import configure_logger
    configure_logger('neuralnilm.log')
    eval("{next_job}.run('{next_job}')".format(next_job=next_job))


def _delete_completed_job():
    with open(job_list_filename, 'r') as fh:
        remaining_jobs = fh.readlines()[1:]
    with open(job_list_filename, 'w') as fh:
        fh.writelines(remaining_jobs)
    logger.info("Remaining jobs = {}".format(remaining_jobs))


if __name__ == "__main__":
    main()
