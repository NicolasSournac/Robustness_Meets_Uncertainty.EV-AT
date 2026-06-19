from typing import Annotated

from pydantic import Field

Epoch = Annotated[int, Field(..., ge=0)]
