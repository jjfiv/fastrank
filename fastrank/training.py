import attr
from typing import Union, Any, Dict
import random
from .clib import CQRel, query_json


@attr.s
class CoordinateAscentParams(object):
    """
    This class represents the configuration and parameters available for FastRank's CoordinateAscent Model.
    """

    num_restarts = attr.ib(type=int, default=5)
    num_max_iterations = attr.ib(type=int, default=25)
    step_base = attr.ib(type=float, default=0.05)
    step_scale = attr.ib(type=float, default=2.0)
    tolerance = attr.ib(type=float, default=0.001)
    normalize = attr.ib(type=bool, default=True)
    init_random = attr.ib(type=bool, default=True)
    output_ensemble = attr.ib(type=bool, default=False)
    seed = attr.ib(type=int, default=random.randint(0, (1 << 64) - 1))
    quiet = attr.ib(type=bool, default=False)

    def name(self):
        return "CoordinateAscent"

    def to_dict(self):
        return attr.asdict(self, recurse=True)

    @staticmethod
    def from_dict(params) -> "CoordinateAscentParams":
        return CoordinateAscentParams(**params)


@attr.s
class RandomForestParams(object):
    """
    This class represents the configuration and parameters available for FastRank's Random Forest Model.
    """

    num_trees = attr.ib(type=int, default=100)
    weight_trees = attr.ib(type=bool, default=True)
    split_method = attr.ib(type=str, default="SquaredError")
    instance_sampling_rate = attr.ib(type=float, default=0.5)
    feature_sampling_rate = attr.ib(type=float, default=0.25)
    min_leaf_support = attr.ib(type=int, default=10)
    split_candidates = attr.ib(type=int, default=3)
    max_depth = attr.ib(type=int, default=8)
    seed = attr.ib(type=int, default=random.randint(0, (1 << 64) - 1))
    quiet = attr.ib(type=bool, default=False)

    def name(self):
        return "RandomForest"

    def to_dict(self):
        return attr.asdict(self, recurse=True)

    @staticmethod
    def from_dict(params) -> "RandomForestParams":
        return RandomForestParams(**params)


@attr.s
class TrainRequest(object):
    """
    This class represents the configuration and parameters available to train a model.

    Models available:

    - `~coordinate_ascent`
    - `~random_forest`
    """

    measure = attr.ib(type=str, default="ndcg")
    params= attr.ib(type=Union[CoordinateAscentParams, RandomForestParams],
        factory=CoordinateAscentParams
    )
    judgments = attr.ib(type=CQRel, default=None)

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
    def from_dict(params) -> "TrainRequest":
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
            params = RandomForestParams.from_dict(params_dict["RandomForest"])
        elif "CoordinateAscent" in params_dict:
            params = CoordinateAscentParams.from_dict(params_dict["CoordinateAscent"])
        else:
            raise ValueError(
                "Python doesn't know about model-params: {}".format(params_dict)
            )
        return TrainRequest(measure, params, judgments)
