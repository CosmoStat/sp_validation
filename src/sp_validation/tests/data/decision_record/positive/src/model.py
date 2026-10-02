# ruff: noqa: F821, F841
# This is parsed, never imported: its unresolved calls prove static reading.
# @sc [decision:file_span,scope:file]
# The whole file supplies this value.
final_value = 11

# @sc [decision:statement,label:selection] module-contract
# Keep the configured count fixed.
module_value = 8


def fit(
    threshold=0.0005, iterations=100, *, enabled=True, optional=None, limits=[1, 250]
):
    """Fit the catalogue.

    @sc [decision:defaults,decision:body] fit-contract
    The selected settings must remain coupled.
    """
    assigned = 1
    configure(keyword=3)
    values = {"dict_entry": 4}
    more = dict(dict_call=5)
    self.attr = 7
    return configure(returned=6)


def nested():
    # @sc [decision:nested]
    local_value = 9
    unrelated = 1000
    return local_value


class Shapes:
    """Shape model.

    @sc [decision:class_span]
    """

    class_value = 10


# @sc [decision:calibration.response,decision:calibration.validation.noise]
settings = {"response": 0.01, "noise": 0.02}
