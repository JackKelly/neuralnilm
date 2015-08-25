from __future__ import print_function, division
import pytest


class TestSyntheticAggregateSource:
    @pytest.fixture
    def synthetic_aggregate_source(self):
        from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
        return SyntheticAggregateSource(
            activations={
                'train': {
                    'fridge': {
                        'UK-DALE_building_1': ['a'] * 10,
                        'Tracebase_building_1': ['b'] * 1
                    },
                    'kettle': {'UK-DALE_building_1': []},
                    'toaster': {'UK-DALE_building_1': []}
                }
            },
            target_appliance='kettle',
            seq_length=128
        )

    def test_distrator_appliances(self, synthetic_aggregate_source):
        source = synthetic_aggregate_source
        distractors = set(source._distractor_appliances(fold='train'))
        assert distractors == set(['fridge', 'toaster'])

    def get_number_of_activations(self, source):
        n_a = 0
        n_b = 0
        for i in range(1000):
            activation = source._select_activation(
                fold='train', appliance='fridge')
            if activation == 'a':
                n_a += 1
            elif activation == 'b':
                n_b += 1
        return n_a, n_b

    def test_select_activation(self,  synthetic_aggregate_source):
        source = synthetic_aggregate_source
        source.uniform_prob_of_selecting_each_model = True
        n_a, n_b = self.get_number_of_activations(source)
        assert abs(n_a - n_b) < 100

        source.uniform_prob_of_selecting_each_model = False
        n_a, n_b = self.get_number_of_activations(source)
        assert (n_a - n_b) > 600

        with pytest.raises(RuntimeError):
            source._select_activation(fold='train', appliance='kettle')

    def test_position_activation(self, synthetic_aggregate_source):
        import numpy as np
        source = synthetic_aggregate_source
        source.allow_incomplete_target = True
        activation = np.ones(256, dtype=np.float32)
        mean_accumulator = 0
        N = 100
        for i in range(N):
            positioned_activation, is_complete = source._position_activation(
                activation, is_target_appliance=True)
            assert not is_complete
            assert len(positioned_activation) == source.seq_length
            mean_accumulator += np.mean(positioned_activation)
        mean = mean_accumulator / N
        assert mean < 1.0  # check that the activation is sometimes offset

        source.allow_incomplete_target = False
        with pytest.raises(RuntimeError):
            source._position_activation(activation, is_target_appliance=True)
