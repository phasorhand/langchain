"""Test the Run Collector Callback Manager"""

from langchain.callbacks.manager import collect_runs
from langchain.chains.transform import TransformChain


def test_collect_runs():
    chain = TransformChain(
        transform=lambda x: x, input_variables=["foo"], output_variables=["foo"]
    )
    with collect_runs() as cb:
        results = [chain({"foo": 3}, include_run_info=True) for _ in range(3)]
        run_ids = [result["__run"].run_id for result in results]
        assert len(cb.traced_runs) == 3
        assert run_ids == [run.id for run in cb.traced_runs]
