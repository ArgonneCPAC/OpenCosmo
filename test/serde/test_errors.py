import opencosmo.serde as serde
import pytest
from opencosmo.serde import (
    ColumnReference,
    SelectMessage,
    SerdeError,
    SerdeErrorCategory,
    apply_message,
    serialize_result,
)


def test_only_public_serde_functions_are_boundaries():
    public_functions = {
        name
        for name in serde.__all__
        if callable(getattr(serde, name)) and not isinstance(getattr(serde, name), type)
    }

    assert public_functions == {"apply_message", "serialize_result"}


def test_serialize_result_returns_structured_error():
    result = serialize_result(object())

    assert isinstance(result, SerdeError)
    assert result.kind == "error"
    assert result.operation == "serialize_result"
    assert result.category is SerdeErrorCategory.UNSUPPORTED_TYPE
    assert result.exception_type == "TypeError"
    assert result.input_type == "object"
    assert result.model_dump(mode="json")["kind"] == "error"


def test_serialize_result_captures_uuid_resolver_failure(test_data):
    dataset = __import__("opencosmo").open(test_data.snapshot.primary.halo_properties)

    def fail(_value):
        raise RuntimeError("resolver failed")

    result = serialize_result(dataset, resolve_uuid=fail)

    assert isinstance(result, SerdeError)
    assert result.category is SerdeErrorCategory.INTERNAL_ERROR
    assert result.message == "resolver failed"


def test_apply_message_returns_structured_error(test_data):
    dataset = __import__("opencosmo").open(test_data.snapshot.primary.halo_properties)
    result = apply_message(
        dataset,
        SelectMessage(columns=("column_that_does_not_exist",)),
    )

    assert isinstance(result, SerdeError)
    assert result.operation == "apply_message"
    assert result.category is SerdeErrorCategory.MISSING_COLUMN
    assert result.target_type == "Dataset"
    assert result.message_type == "SelectMessage"


def test_apply_message_captures_expression_failure(test_data):
    dataset = __import__("opencosmo").open(test_data.snapshot.primary.halo_properties)
    result = apply_message(
        dataset,
        SelectMessage(
            derived_columns={"constant": ColumnReference(name="missing_column")}
        ),
    )

    assert isinstance(result, SerdeError)
    assert result.operation == "apply_message"


def test_public_boundaries_do_not_swallow_base_exceptions(monkeypatch):
    class InterruptingDataset:
        def __len__(self):
            return 0

        @property
        def columns(self):
            return []

        @property
        def descriptions(self):
            return {}

        @property
        def units(self):
            return {}

        @property
        def header(self):
            raise KeyboardInterrupt

    monkeypatch.setattr("opencosmo.serde.summary.Dataset", InterruptingDataset)

    with pytest.raises(KeyboardInterrupt):
        serialize_result(InterruptingDataset())  # type: ignore[arg-type]
