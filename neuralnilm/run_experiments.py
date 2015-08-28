from __future__ import print_function, division
from os.path import expanduser, isdir, exists
import sys
import subprocess
from neuralnilm.config import CONFIG

import logging
logger = logging.getLogger('neuralnilm')


"""
Example contents of config.py:

CONFIG = {
    "MONGODB_ADDRESS": "",

    "PATHS": {  # these paths must already exist
        'OUTPUT': '~/temp/neural_nilm/output',  # save weights & activations
        'EXPERIMENT_DEFINITIONS': '~/temp/neural_nilm/experiment_definitions'
    },

    "JOB_LIST": "~/temp/neural_nilm/job_list.txt"
}
"""


def main():
    _expand_and_check_paths()
    _expand_and_check_job_list()
    print("CONFIG =")
    _print_dict(CONFIG)

    exp_path = CONFIG['PATHS']['EXPERIMENT_DEFINITIONS']
    if exp_path not in sys.path:
        sys.path.insert(0, exp_path)

    next_job = _get_next_job()
    while next_job:
        _run_job(next_job)
        _delete_completed_job()
        next_job = _get_next_job()


def _get_next_job():
    with open(CONFIG['JOB_LIST'], 'r') as fh:
        next_job = fh.readline().strip()
    return next_job


def _run_job(next_job):
    logger.info("Running {}".format(next_job))
    exec("import {:s}".format(next_job))
    eval("{next_job}.run('{next_job}')".format(next_job=next_job))


def _delete_completed_job():
    with open(CONFIG['JOB_LIST'], 'r') as fh:
        remaining_jobs = fh.readlines()[1:]
    with open(CONFIG['JOB_LIST'], 'w') as fh:
        fh.writelines(remaining_jobs)
    logger.info("Remaining jobs = {}".format(remaining_jobs))


def _expand_and_check_paths():
    PATHS = CONFIG['PATHS']
    for key in PATHS:
        PATHS[key] = expanduser(PATHS[key])
        path = PATHS[key]
        if not exists(path):
            raise RuntimeError("Directory '{}' = '{}' does not exist!"
                               .format(key, path))
        if not isdir(path):
            raise RuntimeError("'{}' = '{}' is not a directory!"
                               .format(key, path))


def _expand_and_check_job_list():
    CONFIG['JOB_LIST'] = expanduser(CONFIG['JOB_LIST'])
    if not exists(CONFIG['JOB_LIST']):
        raise RuntimeError(
            "job list '{}' does not exist!".format(CONFIG['JOB_LIST']))


def _print_dict(dictionary, indent=0):
    indent_str = "  " * indent
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print(indent_str, key, "=")
            _print_dict(value, indent=indent+1)
        else:
            print(indent_str, key, "=", "'{}'".format(value))


if __name__ == "__main__":
    main()
