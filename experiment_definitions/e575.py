from __future__ import print_function, division
import os

from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer

from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy
from neuralnilm.net import Net
from neuralnilm.trainer import Trainer
from neuralnilm.metrics import Metrics
from neuralnilm.config import CONFIG


NILMTK_FILENAME = '/data/mine/vadeec/merged/ukdale.h5'
SEQ_LENGTHS = {'kettle': 256}
SAMPLE_PERIOD = 6
STRIDE = 4
APPLIANCES = ['kettle', 'microwave', 'washing machine']


LOADER_CONFIG = {
    'nilmtk_activations': dict(
        appliances=APPLIANCES,
        filename=NILMTK_FILENAME,
        sample_period=SAMPLE_PERIOD,
        windows={
            'train': {
                1: ("2014-01-01", "2014-02-01")
            },
            'unseen_activations_of_seen_appliances': {
                1: ("2014-02-02", "2014-02-08")
            },
            'unseen_appliances': {
                2: ("2013-06-01", "2013-06-07")
            }
        }
    )
}


def run(root_experiment_name):
    activations = get_activations()
    for get_net in [ae]:
        for target_appliance in ['kettle']:
            pipeline = get_pipeline(activations, target_appliance)

            # Build net
            batch = pipeline.get_batch()
            net = get_net(batch)

            # Trainer
            trainer = Trainer(
                net=net,
                data_pipeline=pipeline,
                experiment_id=[
                    root_experiment_name, get_net.__name__, target_appliance],
                metrics=Metrics(state_boundaries=[4]),
                learning_rates={0: 1E-2},
                repeat_callbacks=[
                    (100, Trainer.validate)
                ]
            )

            """
            trainer.submit_initial_report(
                additional_report_contents={
                    'activations': LOADER_CONFIG
                    }
            )
            """

            # Run!
            trainer.fit(1000)


# -------------  NETWORKS  ---------------
def ae(batch):
    input_shape = batch.after_processing.input.shape
    target_shape = batch.after_processing.target.shape

    input_layer = InputLayer(
        shape=input_shape
    )

    # Dense layers
    NUM_UNITS = {
        'dense_layer_0': 100,
        'dense_layer_1':  50,
        'dense_layer_2': 100
    }
    dense_layer_0 = DenseLayer(
        input_layer,
        num_units=NUM_UNITS['dense_layer_0']
    )
    dense_layer_1 = DenseLayer(
        dense_layer_0,
        num_units=NUM_UNITS['dense_layer_1']
    )
    dense_layer_2 = DenseLayer(
        dense_layer_1,
        num_units=NUM_UNITS['dense_layer_2']
    )

    # Output
    final_dense_layer = DenseLayer(
        dense_layer_2,
        num_units=target_shape[1] * target_shape[2],
        nonlinearity=None
    )
    output_layer = ReshapeLayer(
        final_dense_layer,
        shape=target_shape
    )

    net = Net(output_layer)
    return net


def get_activations():
    nilmtk_activations = load_nilmtk_activations(
        **LOADER_CONFIG['nilmtk_activations'])
    return nilmtk_activations


def get_pipeline(activations, target_appliance):
    source = SyntheticAggregateSource(
        activations=activations,
        target_appliance=target_appliance,
        seq_length=SEQ_LENGTHS[target_appliance],
        allow_incomplete_target=False
    )

    sample = source.get_batch(num_seq_per_batch=1024)
    input_std = sample.before_processing.input.flatten().std()
    target_std = sample.before_processing.target.flatten().std()
    pipeline = DataPipeline(
        [source],
        num_seq_per_batch=64,
        input_processing=[DivideBy(input_std)],
        target_processing=[DivideBy(target_std)]
    )

    return pipeline
