# neuralnilm
Deep Neural Networks Applied to Energy Disaggregation

This is a complete re-write of the
[Neural NILM Prototype](https://github.com/JackKelly/neuralnilm_prototype).

# Getting started
This is 'research' software and will be undergoing a lot of changes
over the coming weeks and months!  Be warned!  Also, my efforts will
be focused on writing code, so I probably won't have time to look at
the issue queue any time soon so please consider this software to
essentially be unsupported for the time being.

That said, if you're really determined to read and understand this
code then here's a quite description of the basic architecture:

Each 'experiment' is defined in a Python script in
`experiment_definitions/`.  Each of these files must have a `run`
function which, well, runs the experiment!

`experiment_definitions/` also should include a text file called
`job_list.txt` which is just a list of experiment names like `e575`.
The script `scripts/run_experiments.py` looks at `job_list.txt` to
find the next job to run.  After each job completes,
`run_experiments.py` removes that job from the `job_list`.  In this
way, it is possible to add, remove or re-order jobs from the
`job_list` while an experiment is running.

The re-usable code which makes up the bulk of the NeuralNILM package
lives in `neuralnilm/neuralnilm`.  `net.py` includes a `Net` class
which handles the construction of each neural network.  `trainer.py`
includes a `Trainer` class which trains each net and sends metrics to
the Mongo database.

`neuralnilm/neuralnilm/data` includes the mechanisms for loading and
transforming data.  This is designed to be very modular so you can
easily mix and match data sources, pre-processing steps and networks.
For a quick overview of how this all works, take a look at some of the
python files in the `experiment_definitions` directory.

`neuralnilm/neuralnilm/monitor` contains the mechanisms for loading
metrics and metadata from the Mongo database and plotting it.


# Requirements
See `requirements.txt` for the required Python packages.  Also requires Mongo DB.

# Configuration
You need to create a `~/.neuralnilm` text file to specify some
parameters.  Please see `example_config_file`.

## MongoDB remote access

See http://www.mkyong.com/mongodb/mongodb-allow-remote-access/
