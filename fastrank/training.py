from typing import Union, Any, Dict, Optional
import random
from .clib import CQRel, query_json
from dataclasses import dataclass, asdict


@dataclass
class CoordinateAscentParams:
    """
    This class represents the configuration and parameters available for FastRank's CoordinateAscent Model.
    """

    num_restarts: int = 5
    num_max_iterations: int = 25
    step_base: float = 0.05
    step_scale: float = 2.0
    tolerance: float = 0.001
    normalize: bool = True
    init_random: bool = True
    output_ensemble: bool = False
    seed: int = random.randint(0, (1 << 64) - 1)
    quiet: bool = False

    def name(self):
        return "CoordinateAscent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(params: Dict[str, Any]) -> "CoordinateAscentParams":
        return CoordinateAscentParams(**params)


@dataclass
class RandomForestParams:
    """
    This class represents the configuration and parameters available for FastRank's Random Forest Model.
    """

    num_trees: int = 100
    weight_trees: bool = True
    split_method: str = "SquaredError"
    instance_sampling_rate: float = 0.5
    feature_sampling_rate: float = 0.25
    min_leaf_support: int = 10
    split_candidates: int = 3
    max_depth: int = 8
    seed: int = random.randint(0, (1 << 64) - 1)
    quiet: bool = False

    def name(self):
        return "RandomForest"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(params: Dict[str, Any]) -> "RandomForestParams":
        return RandomForestParams(**params)


@dataclass
class TrainRequest:
    """
    This class represents the configuration and parameters available to train a model.

    Models available:

    - `~coordinate_ascent`
    - `~random_forest`
    """

    measure: str = "ndcg"
    params: Union[CoordinateAscentParams, RandomForestParams] = CoordinateAscentParams()
    judgments: Optional[CQRel] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert into an untyped, JSON-safe representation from a TrainRequest.
        """
        wrapped_params = {self.params.name(): self.params.to_dict()}
        judgments = None
        if self.judgments is not None:
            judgments = self.judgments.to_dict()
        return {
            "measure": self.measure,
            "params": wrapped_params,
            "judgments": judgments,
        }

    def clone(self) -> "TrainRequest":
        """
        Copy this model's parameters! So you can modify it without messing up someone else.
        """
        return TrainRequest.from_dict(self.to_dict())

    @staticmethod
    def coordinate_ascent() -> "TrainRequest":
        """
        Get a new TrainRequest with default parameters for a Coordinate Ascent Model.
        """
        return TrainRequest.from_dict(query_json("coordinate_ascent_defaults"))

    @staticmethod
    def random_forest() -> "TrainRequest":
        """
        Get a new TrainRequest with default parameters for a Random Forest Model.
        """
        return TrainRequest.from_dict(query_json("random_forest_defaults"))

    @staticmethod
    def from_dict(params: Dict[str, Any]) -> "TrainRequest":
        """
        From an untyped, JSON-safe representation into a TrainRequest.
        """
        measure = params["measure"]
        judgments = None
        if params["judgments"] is not None:
            judgments = CQRel.from_dict(params["judgments"])
        params_dict = params["params"]
        if len(params_dict) != 1:
            raise ValueError("What do I do with this?: {}".format(params_dict))
        if "RandomForest" in params_dict:
            cfg = RandomForestParams.from_dict(params_dict["RandomForest"])
        elif "CoordinateAscent" in params_dict:
            cfg = CoordinateAscentParams.from_dict(params_dict["CoordinateAscent"])
        else:
            raise ValueError(
                "Python doesn't know about model-params: {}".format(params_dict)
            )
        return TrainRequest(measure, cfg, judgments)
