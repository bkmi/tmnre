from typing import Dict, Tuple, Union

from swyft.types import Array

MarginalKey = Union[Tuple[int], Tuple[str]]
Marginals = Dict[MarginalKey, Array]
