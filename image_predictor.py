import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


DEFAULT_CLASS_LABELS = [
    ("almirah", "수납장"),
    ("chair", "의자"),
    ("fridge", "냉장고"),
    ("table", "테이블"),
    ("tv", "TV"),
]

DEFAULT_IMAGE_SIZE = (160, 160)
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


class FurnitureImagePredictor:
    def __init__(self, model_path=None, class_names_path=None):
        base_dir = Path(__file__).resolve().parent
        self.model_path = self._find_existing_path(
            [
                Path(model_path) if model_path else None,
                base_dir / "furniture_transfer_160_compat.keras",
                base_dir / "furniture_transfer_160_best_compat.keras",
                base_dir / "furniture_transfer_160.keras",
                base_dir / "furniture_transfer_160_best.keras",
                base_dir / "models" / "furniture_transfer_160_compat.keras",
                base_dir / "models" / "furniture_transfer_160_best_compat.keras",
                base_dir / "models" / "furniture_transfer_160.keras",
                base_dir / "models" / "furniture_transfer_160_best.keras",
            ],
            "model",
        )
        self.class_names_path = self._find_optional_path(
            [
                Path(class_names_path) if class_names_path else None,
                base_dir / "class_names.json",
                base_dir / "models" / "class_names.json",
            ]
        )
        self.class_labels, self.image_size = self._load_class_labels()
        self.model = load_model(self.model_path)

    def _find_existing_path(self, candidates, target_name):
        for path in candidates:
            if path and path.exists():
                return path
        searched = "\n".join(str(path) for path in candidates if path)
        raise FileNotFoundError(f"{target_name} file not found. searched paths:\n{searched}")

    def _find_optional_path(self, candidates):
        for path in candidates:
            if path and path.exists():
                return path
        return None

    def _load_class_labels(self):
        if not self.class_names_path:
            return DEFAULT_CLASS_LABELS, DEFAULT_IMAGE_SIZE

        data = json.loads(self.class_names_path.read_text(encoding="utf-8"))
        class_names_en = data.get("class_names_en", [])
        class_names_ko = data.get("class_names_ko", [])
        image_size = tuple(data.get("image_size", DEFAULT_IMAGE_SIZE))

        if len(class_names_en) != len(class_names_ko) or not class_names_en:
            return DEFAULT_CLASS_LABELS, DEFAULT_IMAGE_SIZE

        return list(zip(class_names_en, class_names_ko)), image_size

    def _open_image(self, image_file):
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        return Image.open(image_file).convert("RGB")

    def _resize_direct(self, image):
        return image.resize(self.image_size, RESAMPLE)

    def _resize_square(self, image):
        square_size = max(image.size)
        canvas = Image.new("RGB", (square_size, square_size), (255, 255, 255))
        paste_x = (square_size - image.width) // 2
        paste_y = (square_size - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
        return canvas.resize(self.image_size, RESAMPLE)

    def _center_crop(self, image):
        crop_size = min(image.size)
        left = (image.width - crop_size) // 2
        top = (image.height - crop_size) // 2
        return image.crop((left, top, left + crop_size, top + crop_size)).resize(
            self.image_size,
            RESAMPLE,
        )

    def preprocess(self, image_file):
        image = self._open_image(image_file)
        original_size = image.size

        direct = self._resize_direct(image)
        square = self._resize_square(image)
        cropped = self._center_crop(image)
        variants = [
            direct,
            ImageOps.mirror(direct),
            square,
            cropped,
        ]

        # The transfer-learning model already contains MobileNetV2 preprocessing,
        # so the app passes raw 0-255 RGB values instead of dividing by 255.
        image_array = np.asarray(variants, dtype=np.float32)
        return image_array, original_size

    def predict(self, image_file):
        image_array, original_size = self.preprocess(image_file)
        batch_probabilities = self.model.predict(image_array, verbose=0)
        probabilities = batch_probabilities.mean(axis=0)
        pred_index = int(np.argmax(probabilities))
        class_en, class_ko = self.class_labels[pred_index]

        return {
            "class_en": class_en,
            "class_ko": class_ko,
            "confidence": float(probabilities[pred_index]),
            "original_size": original_size,
            "image_size": self.image_size,
            "preprocess_mode": "multi_view_average",
            "preprocess_count": int(len(image_array)),
            "probabilities": [
                {
                    "category_en": label_en,
                    "category_ko": label_ko,
                    "probability": float(prob),
                }
                for (label_en, label_ko), prob in zip(self.class_labels, probabilities)
            ],
        }
