from pydantic import TypeAdapter

from ev_at.core.pydantic.annotated_type import Epoch


def validate_epoch(value: int) -> Epoch:
    return TypeAdapter(Epoch).validate_python(value)
