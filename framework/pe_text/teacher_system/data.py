from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from . import NUM_ATTRIBUTES


@dataclass(frozen=True)
class AnnotationTable:
    attribute_names: tuple[str, ...]
    image_paths: tuple[str, ...]
    labels: np.ndarray


def read_annotation_csv(path: str | Path) -> AnnotationTable:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise ValueError(f"Empty annotation file: {path}") from exc

        if not header or not header[0].startswith("#"):
            raise ValueError(f"Expected '# image' header in {path}")
        attribute_names = tuple(name.strip() for name in header[1:])
        if len(attribute_names) != NUM_ATTRIBUTES:
            raise ValueError(
                f"Expected {NUM_ATTRIBUTES} attributes in {path}, got {len(attribute_names)}"
            )

        image_paths: list[str] = []
        labels: list[list[float]] = []
        for line_number, row in enumerate(rows, start=2):
            if not row:
                continue
            if len(row) != NUM_ATTRIBUTES + 1:
                raise ValueError(
                    f"Expected {NUM_ATTRIBUTES + 1} columns at {path}:{line_number}, "
                    f"got {len(row)}"
                )
            image_paths.append(row[0])
            labels.append([float(value) for value in row[1:]])

    return AnnotationTable(
        attribute_names=attribute_names,
        image_paths=tuple(image_paths),
        labels=np.asarray(labels, dtype=np.float32),
    )


def read_annotation_csvs(paths: Sequence[str | Path]) -> AnnotationTable:
    tables = [read_annotation_csv(path) for path in paths]
    if not tables:
        raise ValueError("At least one annotation CSV is required")
    names = tables[0].attribute_names
    if any(table.attribute_names != names for table in tables[1:]):
        raise ValueError("Annotation CSV attribute orders differ")
    image_paths = tuple(path for table in tables for path in table.image_paths)
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Annotation CSVs contain duplicate image paths")
    return AnnotationTable(
        attribute_names=names,
        image_paths=image_paths,
        labels=np.concatenate([table.labels for table in tables], axis=0),
    )


def read_matrix_csv(path: str | Path, expected_columns: int = NUM_ATTRIBUTES) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows, None)
        if header is None:
            raise ValueError(f"Empty matrix file: {path}")
        data = [[float(value) for value in row] for row in rows if row]
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"Expected (*, {expected_columns}) matrix in {path}, got {matrix.shape}"
        )
    return matrix


def read_retrieval_ids(path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    image_paths: list[str] = []
    semantic_ids: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Expected 2 columns at {path}:{line_number}")
            image_paths.append(row[0])
            semantic_ids.append(int(row[1]))
    return tuple(image_paths), np.asarray(semantic_ids, dtype=np.int64)


class LetterboxSiglipTransform:
    def __init__(
        self,
        image_size: int | tuple[int, int] = 512,
        train: bool = False,
        flip_probability: float = 0.5,
    ) -> None:
        # int -> square canvas; (height, width) -> rectangular canvas
        self.height, self.width = (image_size, image_size) if isinstance(image_size, int) else (int(image_size[0]), int(image_size[1]))
        self.image_size = image_size
        self.train = train
        self.flip_probability = flip_probability

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.train and random.random() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        width, height = image.size
        scale = min(self.width / width, self.height / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.width, self.height), (128, 128, 128))
        offset = (
            (self.width - resized.width) // 2,
            (self.height - resized.height) // 2,
        )
        canvas.paste(resized, offset)

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5


class LowQualitySiglipTransform(LetterboxSiglipTransform):
    """Conservative train-only camera degradation without changing color labels.

    Exactly one degradation is sampled per affected image. The validation and
    submission paths continue to use the unchanged letterbox transform.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = (256, 128),
        degrade_probability: float = 0.5,
        flip_probability: float = 0.5,
    ) -> None:
        super().__init__(image_size=image_size, train=True, flip_probability=flip_probability)
        if not 0.0 <= degrade_probability <= 1.0:
            raise ValueError("degrade_probability must be in [0, 1]")
        self.degrade_probability = degrade_probability

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if random.random() < self.degrade_probability:
            operation = random.randrange(4)
            if operation == 0:
                # Simulate a smaller source crop, then restore its original size.
                scale = random.uniform(0.5, 0.85)
                small = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.Resampling.BICUBIC,
                )
                image = small.resize(image.size, Image.Resampling.BICUBIC)
            elif operation == 1:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))
            elif operation == 2:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=random.randint(45, 80))
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    image = compressed.convert("RGB")
            else:
                array = np.asarray(image, dtype=np.float32)
                noise = np.random.normal(0.0, random.uniform(2.0, 5.0), array.shape)
                image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
        return super().__call__(image)


class SurveillanceTransform(LetterboxSiglipTransform):
    """Train-only surveillance degradation, followed by the legacy letterbox.

    Operations have independent probabilities and run on the source crop before
    resizing. No hue, saturation, white-balance or grayscale transform is used.
    Seed Python and NumPy RNGs for reproducible results, as in the training CLI.
    """

    probabilities = (0.5, 0.5, 0.3, 0.5, 0.5, 0.2, 0.2)

    def __init__(
        self,
        image_size: int | tuple[int, int] = (256, 128),
        flip_probability: float = 0.5,
    ) -> None:
        super().__init__(image_size=image_size, train=True, flip_probability=flip_probability)

    @staticmethod
    def bbox_crop(image: Image.Image) -> Image.Image:
        width = max(1, round(image.width * random.uniform(0.88, 1.0)))
        height = max(1, round(image.height * random.uniform(0.88, 1.0)))
        left = random.randint(0, image.width - width)
        top = random.randint(0, image.height - height)
        return image.crop((left, top, left + width, top + height))

    @staticmethod
    def low_resolution(image: Image.Image) -> Image.Image:
        target_height = random.randint(48, 128)
        if image.height <= target_height:
            return image
        downsample = random.choice((Image.Resampling.BILINEAR,
                                    Image.Resampling.BICUBIC, Image.Resampling.NEAREST))
        upsample = random.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC))
        small = image.resize((max(1, round(image.width * target_height / image.height)),
                              target_height), downsample)
        return small.resize(image.size, upsample)

    @staticmethod
    def motion_blur(image: Image.Image, length: int, angle: float) -> Image.Image:
        """Average bilinearly sampled offsets along a straight line; edge padding."""
        if not 3 <= length <= 9:
            raise ValueError("motion blur length must be in [3, 9]")
        array = np.asarray(image, dtype=np.float32)
        radius = (length - 1) / 2.0
        pad = int(np.ceil(radius)) + 1
        padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        result = np.zeros_like(array)
        radians = np.deg2rad(angle)
        for offset in np.linspace(-radius, radius, length):
            dx, dy = offset * np.cos(radians), offset * np.sin(radians)
            x, y = int(np.floor(dx)), int(np.floor(dy))
            fx, fy = dx - x, dy - y
            for sx, sy, weight in ((x, y, (1-fx)*(1-fy)), (x+1, y, fx*(1-fy)),
                                   (x, y+1, (1-fx)*fy), (x+1, y+1, fx*fy)):
                if weight > 0:
                    result += weight * padded[pad+sy:pad+sy+image.height,
                                              pad+sx:pad+sx+image.width]
        return Image.fromarray(np.clip(np.rint(result / length), 0, 255).astype(np.uint8))

    @staticmethod
    def optical(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
        return ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))

    @staticmethod
    def small_occlusion(image: Image.Image) -> Image.Image:
        area = image.width * image.height * random.uniform(0.02, 0.08)
        ratio = random.uniform(0.5, 2.0)  # width / height
        width, height = max(1, round(np.sqrt(area * ratio))), max(1, round(np.sqrt(area / ratio)))
        # Skip an infeasible rectangle on very narrow crops, rather than making
        # an oversized mask or changing its requested aspect ratio.
        if width > image.width or height > image.height:
            return image
        left = random.randint(0, image.width - width)
        top = random.randint(0, image.height - height)
        image = image.copy()
        image.paste((128, 128, 128), (left, top, left + width, top + height))
        return image

    def augment_crop(self, image: Image.Image) -> Image.Image:
        """Degrade once before rendering; flip is handled by the caller."""
        image = image.convert("RGB")
        if random.random() < self.probabilities[0]:
            image = self.bbox_crop(image)
        if random.random() < self.probabilities[1]:
            image = self.low_resolution(image)
        if random.random() < self.probabilities[2]:
            if random.randrange(2) == 0:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
            else:
                image = self.motion_blur(image, random.randint(3, 9), random.uniform(0, 180))
        if random.random() < self.probabilities[3]:
            image = self.optical(image)
        if random.random() < self.probabilities[4]:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(20, 60))
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                image = compressed.convert("RGB")
        if random.random() < self.probabilities[5]:
            array = np.asarray(image, dtype=np.float32)
            noise = np.random.normal(0.0, random.uniform(2.0, 6.0), array.shape)
            image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
        if random.random() < self.probabilities[6]:
            image = self.small_occlusion(image)
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return super().__call__(self.augment_crop(image))


class Surveillance2Transform(SurveillanceTransform):
    """B1: same operations, 1.5x probabilities and 32--96px low resolution."""

    probabilities = tuple(min(0.9, p * 1.5) for p in SurveillanceTransform.probabilities)

    @staticmethod
    def low_resolution(image: Image.Image) -> Image.Image:
        target_height = random.randint(32, 96)
        if image.height <= target_height:
            return image
        downsample = random.choice((Image.Resampling.BILINEAR,
                                    Image.Resampling.BICUBIC, Image.Resampling.NEAREST))
        upsample = random.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC))
        small = image.resize((max(1, round(image.width * target_height / image.height)),
                              target_height), downsample)
        return small.resize(image.size, upsample)


class PairedTeacherTransform:
    """A1: one crop/degradation/flip, two deterministic letterbox canvases."""

    def __init__(self, student_hw=(256, 128), teacher_hw=(256, 256),
                 profile="surveillance", augment=True, flip_probability=0.5):
        if profile not in ("surveillance", "surveillance2"):
            raise ValueError("Online teacher requires surveillance or surveillance2")
        self.degrader = (SurveillanceTransform if profile == "surveillance"
                         else Surveillance2Transform)(student_hw)
        self.student = LetterboxSiglipTransform(student_hw)
        self.teacher = LetterboxSiglipTransform(teacher_hw)
        self.augment = augment
        self.flip_probability = flip_probability

    def __call__(self, image):
        crop = self.degrader.augment_crop(image) if self.augment else image.convert("RGB")
        if self.augment and random.random() < self.flip_probability:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.student(crop), self.teacher(crop)


class PedestrianAttributeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        annotation_csv: str | Path,
        data_root: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.table = read_annotation_csv(annotation_csv)
        self.data_root = Path(data_root)
        self.transform = transform or LetterboxSiglipTransform()

    def __len__(self) -> int:
        return len(self.table.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        relative_path = self.table.image_paths[index]
        image_path = self.data_root / Path(relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing dataset image: {image_path}")
        with Image.open(image_path) as image:
            pixels = self.transform(image)
        label = torch.from_numpy(self.table.labels[index].copy())
        return pixels, label, relative_path


class MultiCsvPedestrianAttributeDataset(PedestrianAttributeDataset):
    def __init__(
        self,
        annotation_csvs: Sequence[str | Path],
        data_root: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.table = read_annotation_csvs(annotation_csvs)
        self.data_root = Path(data_root)
        self.transform = transform or LetterboxSiglipTransform()


def ensure_same_image_order(expected: Sequence[str], actual: Sequence[str]) -> None:
    if tuple(expected) != tuple(actual):
        raise ValueError("Image order mismatch between predictions and retrieval ids")
