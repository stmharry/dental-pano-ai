import dataclasses
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from absl import app, flags, logging
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from PIL import Image
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.ops import scale_masks

flags.DEFINE_multi_string("input", "./data/*", "Input file paths or glob patterns.")
flags.DEFINE_string("output", "./results", "Output directory.")
flags.DEFINE_string(
    "deeplab_config", "./models/deeplab/config.yaml", "DeepLab config file."
)
flags.DEFINE_string(
    "deeplab_weights", "./models/deeplab/model.pth", "DeepLab weights file."
)
flags.DEFINE_string("yolo_config", "./models/yolo/config.yaml", "YOLO config file.")
flags.DEFINE_string("yolo_weights", "./models/yolo/model.pt", "YOLO weights file.")
flags.DEFINE_string("device", "cpu", "Device to use.")
FLAGS = flags.FLAGS


# semantic segmentation


@dataclasses.dataclass(kw_only=True)
class SemanticSegmentationPrediction(object):
    prob: np.ndarray


@dataclasses.dataclass(kw_only=True)
class SemanticSegmentationModule(object):
    config_path: str
    weights_path: str
    device: str = "cpu"

    predictor: DefaultPredictor = dataclasses.field(init=False)

    def __post_init__(self):
        cfg: CfgNode = get_cfg()
        add_deeplab_config(cfg)
        cfg.merge_from_file(self.config_path)

        cfg.DATASETS.TEST = ()
        cfg.INPUT.FORMAT = "RGB"
        cfg.INPUT.CROP.ENABLED = False
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.SOLVER.IMS_PER_BATCH = 1

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image: np.ndarray) -> SemanticSegmentationPrediction:
        prob: np.ndarray = self.predictor(image)["sem_seg"].softmax(dim=0).numpy()

        return SemanticSegmentationPrediction(prob=prob)


# instance detection

IntersectionMetric: TypeAlias = Literal[
    "intersection_over_union",
    "intersection_over_minimum",
]


@dataclasses.dataclass(kw_only=True)
class InstanceDetectionPredictionInstance(object):
    score: float
    category_id: int
    category_name: str
    bbox_xyxy: np.ndarray
    mask: np.ndarray

    @property
    def bbox_xywh(self) -> np.ndarray:
        xb, yb, xe, ye = self.bbox_xyxy
        return np.array([(xb + xe) // 2, (yb + ye) // 2, xe - xb, ye - yb])

    @property
    def bbox_slices(self) -> tuple[slice, ...]:
        xb, yb, xe, ye = self.bbox_xyxy
        return (slice(yb, ye + 1), slice(xb, xe + 1))

    @property
    def mask_area(self) -> int:
        return self.mask[self.bbox_slices].sum()

    def intersect(
        self,
        other: "InstanceDetectionPredictionInstance",
        method: IntersectionMetric,
        epsilon: float = 1e-5,
    ) -> float:

        xb1, yb1, xe1, ye1 = self.bbox_xyxy
        xb2, yb2, xe2, ye2 = other.bbox_xyxy

        bbox_intersection_slices: tuple[slice, ...] = (
            slice(max(yb1, yb2), min(ye1, ye2) + 1),
            slice(max(xb1, xb2), min(xe1, xe2) + 1),
        )
        if self.mask[bbox_intersection_slices].size == 0:
            return 0.0

        bbox_union_slices: tuple[slice, ...] = (
            slice(min(yb1, yb2), max(ye1, ye2) + 1),
            slice(min(xb1, xb2), max(xe1, xe2) + 1),
        )

        mask1: np.ndarray = self.mask[bbox_union_slices]
        mask2: np.ndarray = other.mask[bbox_union_slices]
        intersection: np.ndarray = np.logical_and(mask1, mask2)

        mask1_area: int = mask1.sum()
        mask2_area: int = mask2.sum()
        intersection_area: int = intersection.sum()

        match method:
            case "intersection_over_union":
                return intersection_area / (
                    mask1_area + mask2_area - intersection_area + epsilon
                )

            case "intersection_over_minimum":
                return intersection_area / (min(mask1_area, mask2_area) + epsilon)

    def mask_array(self, other: np.ndarray) -> np.ndarray:
        if self.mask.shape != other.shape[-2:]:
            raise ValueError(
                f"Shape incompatible: {other.shape} v.s. {self.mask.shape}"
            )

        other = other[(..., *self.bbox_slices)]
        mask = self.mask[self.bbox_slices]

        return other[..., mask]


@dataclasses.dataclass(kw_only=True)
class InstanceDetectionPrediction(object):
    instances: list[InstanceDetectionPredictionInstance]

    @classmethod
    def from_ultralytics_results(
        cls, results: Results
    ) -> "InstanceDetectionPrediction":
        assert results.boxes is not None
        assert results.masks is not None

        masks = scale_masks(
            results.masks.data[None, ...], shape=results.orig_shape
        ).squeeze(dim=0)

        instances: list[InstanceDetectionPredictionInstance] = [
            InstanceDetectionPredictionInstance(
                score=float(score),
                category_id=int(category_id),
                category_name=results.names[int(category_id)],
                bbox_xyxy=bbox_xyxy.numpy().astype(int),
                mask=mask.numpy().astype(bool),
            )
            for score, category_id, bbox_xyxy, mask in zip(
                results.boxes.conf,
                results.boxes.cls,
                results.boxes.xyxy,
                masks,
            )
        ]

        return cls(instances=instances)

    def sort_by_score(
        self, direction: Literal["ASCENDING", "DESCENDING"] = "DESCENDING"
    ) -> "InstanceDetectionPrediction":
        return InstanceDetectionPrediction(
            instances=sorted(
                self.instances,
                key=lambda instance: instance.score,
                reverse=(direction == "DESCENDING"),
            ),
        )

    def filter_by_category(self, regex: Any) -> "InstanceDetectionPrediction":
        return InstanceDetectionPrediction(
            instances=[
                instance
                for instance in self.instances
                if re.match(regex, instance.category_name)
            ],
        )

    def filter_by_area(self, threshold: int = 0) -> "InstanceDetectionPrediction":
        return InstanceDetectionPrediction(
            instances=[
                instance
                for instance in self.instances
                if instance.mask_area > threshold
            ],
        )

    def non_maximum_suppression(
        self,
        threshold: float,
        method: IntersectionMetric,
    ) -> "InstanceDetectionPrediction":
        pred: InstanceDetectionPrediction = self.sort_by_score(direction="DESCENDING")

        instances: list[InstanceDetectionPredictionInstance] = pred.instances
        max_instances: list[InstanceDetectionPredictionInstance] = []
        while len(instances) > 0:
            instance: InstanceDetectionPredictionInstance = instances.pop(0)

            _instances: list[InstanceDetectionPredictionInstance] = []
            for other_instance in instances:
                iou = instance.intersect(other_instance, method=method)

                if iou < threshold:
                    _instances.append(other_instance)

            instances = _instances
            max_instances.append(instance)

        return InstanceDetectionPrediction(instances=max_instances)


@dataclasses.dataclass(kw_only=True)
class InstanceDetectionModule(object):
    config_path: str
    weights_path: str

    conf: float = 0.0001
    iou: float = 0.5
    max_det: int = 500
    retina_masks: bool = True
    device: str = "cpu"

    predictor: SegmentationPredictor = dataclasses.field(init=False)

    def __post_init__(self):
        predictor = SegmentationPredictor(
            self.config_path,  # type: ignore
            overrides={
                "mode": "predict",
                "model": self.weights_path,
                "batch": 1,
                "conf": self.conf,
                "iou": self.iou,
                "max_det": self.max_det,
                "retina_masks": self.retina_masks,
                "show": False,
                "save": False,
                "device": self.device,
                "verbose": False,
            },
        )
        predictor.args.embed = None  # type: ignore
        predictor.setup_model(model=None)

        self.predictor = predictor

    def __call__(self, image: np.ndarray) -> InstanceDetectionPrediction:
        results: Results = list(self.predictor(image))[0]  # type: ignore

        return InstanceDetectionPrediction.from_ultralytics_results(results)


# post-processing


class FindingType(str, Enum):
    MISSING = "MISSING"
    IMPLANT = "IMPLANT"
    RESIDUAL_ROOT = "ROOT_REMNANTS"
    CROWN_BRIDGE = "CROWN_BRIDGE"
    ROOT_CANAL_FILLING = "ENDO"
    FILLING = "FILLING"
    CARIES = "CARIES"
    PERIAPICAL_RADIOLUCENCY = "PERIAPICAL_RADIOLUCENT"


@dataclasses.dataclass(kw_only=True)
class FindingEntry(object):
    index: int
    finding: FindingType
    score: float

    @property
    def fdi(self) -> str:
        quadrant: int = (self.index - 1) // 8 + 1
        tooth: int = (self.index - 1) % 8 + 1

        return f"{quadrant}{tooth}"


@dataclasses.dataclass(kw_only=True)
class FindingAssessment(object):
    name: str
    entries: list[FindingEntry]

    def to_csv(self, path: str | Path) -> None:
        pd.DataFrame({
            "file_name": self.name,
            "fdi": [entry.fdi for entry in self.entries],
            "finding": [entry.finding.name for entry in self.entries],
            "score": [entry.score for entry in self.entries],
        }).to_csv(path, index=False)


@dataclasses.dataclass(kw_only=True)
class PostProcessingModule(object):
    nms_method: IntersectionMetric = "intersection_over_minimum"
    nms_threshold: float = 0.3

    def __call__(
        self,
        semseg_pred: SemanticSegmentationPrediction,
        insdet_pred: InstanceDetectionPrediction,
    ) -> list[FindingEntry]:

        semseg_prob: np.ndarray = semseg_pred.prob

        finding_entries: list[FindingEntry] = []
        for finding in FindingType:

            instances: list[InstanceDetectionPredictionInstance]
            match finding:
                case FindingType.MISSING:
                    # NOTE: for tooth objects, additional nms is applied because in
                    #   instance detection we treat each fdi as a separate class
                    instances = (
                        insdet_pred.filter_by_category("^TOOTH")
                        .non_maximum_suppression(
                            method=self.nms_method, threshold=self.nms_threshold
                        )
                        .filter_by_area()
                        .instances
                    )

                case _:
                    instances = (
                        insdet_pred.filter_by_category(f"^{finding.value}$")
                        .filter_by_area()
                        .instances
                    )

            prob_discount: np.ndarray | None = None
            if len(instances):
                prob_discount = np.sum(
                    [instance.mask for instance in instances], axis=0
                ).astype(np.float32)

            finding_scores: list[np.ndarray] = []
            for instance in instances:
                assert prob_discount is not None

                semseg_prob_masked: np.ndarray = instance.mask_array(semseg_prob)
                prob_discount_masked: np.ndarray = instance.mask_array(prob_discount)

                semseg_prob_masked = 1.0 - np.power(
                    1.0 - semseg_prob_masked, 1 / prob_discount_masked
                )
                if semseg_prob_masked.size == 0:
                    continue

                share_per_tooth: np.ndarray = np.mean(semseg_prob_masked, axis=-1)
                share_per_tooth = np.r_[0, share_per_tooth[1:]] / np.sum(
                    share_per_tooth[1:]
                )

                score_per_tooth: np.ndarray
                match finding:
                    case FindingType.MISSING:
                        score_per_tooth = share_per_tooth

                    case _:
                        score_per_tooth = instance.score * share_per_tooth

                finding_scores.append(score_per_tooth)

            finding_score: np.ndarray
            if len(finding_scores) > 0:
                finding_score = 1 - np.prod(1 - np.stack(finding_scores), axis=0)
            else:
                finding_score = np.zeros((len(semseg_prob),), dtype=np.float32)

            match finding:
                case FindingType.MISSING:
                    finding_score = 1 - finding_score

                case _:
                    ...

            for index, _finding_score in enumerate(finding_score):
                # the background class is ignored
                if index == 0:
                    continue

                finding_entry: FindingEntry = FindingEntry(
                    index=index, finding=finding, score=_finding_score
                )
                finding_entries.append(finding_entry)

        return finding_entries


def main(_):
    # module initialization

    semseg_module: SemanticSegmentationModule = SemanticSegmentationModule(
        config_path=FLAGS.deeplab_config,
        weights_path=FLAGS.deeplab_weights,
        device=FLAGS.device,
    )
    insdet_module: InstanceDetectionModule = InstanceDetectionModule(
        config_path=FLAGS.yolo_config,
        weights_path=FLAGS.yolo_weights,
        device=FLAGS.device,
    )
    postproc_module: PostProcessingModule = PostProcessingModule()

    # directory setup

    output_dir: Path = Path(FLAGS.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_paths: list[Path] = []
    for image_path in FLAGS.input:
        if Path(image_path).is_absolute():
            image_paths.extend(
                Path("/").glob(Path(image_path).relative_to("/").as_posix())
            )

        else:
            image_paths.extend(Path(".").glob(image_path))

    # processing

    for image_path in sorted(image_paths):
        csv_path: Path = Path(output_dir, image_path.with_suffix(".csv").name)
        logging.info(f"Processing {image_path}, saving to {csv_path}")

        image_pil: Image.Image = Image.open(image_path).convert("RGB")
        image: np.ndarray = np.asarray(image_pil)

        semseg_pred: SemanticSegmentationPrediction = semseg_module(image)
        insdet_pred: InstanceDetectionPrediction = insdet_module(image)
        finding_entries: list[FindingEntry] = postproc_module(semseg_pred, insdet_pred)

        assessment: FindingAssessment = FindingAssessment(
            name=image_path.stem, entries=finding_entries
        )
        assessment.to_csv(csv_path)


if __name__ == "__main__":
    app.run(main)
