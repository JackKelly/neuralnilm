import pytest


class TestSyntheticAggregateSource:
    @pytest.fixture
    def synthetic_aggregate_source(self):
        from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
        return SyntheticAggregateSource(
            activations={
                'fridge': {'UK-DALE_building_1': []},
                'kettle': {'UK-DALE_building_1': []},
                'toaster': {'UK-DALE_building_1': []}
            },
            target_appliance='kettle',
            seq_length=128
        )

    def test_distrator_appliances(self, synthetic_aggregate_source):
        source = synthetic_aggregate_source
        distractors = set(source._distractor_appliances())
        assert distractors == set(['fridge', 'toaster'])
