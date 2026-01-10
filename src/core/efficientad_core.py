import numpy as np
from typing import Optional, Tuple

class EfficientADCore:
    def __init__(self, teacher, student, autoencoder):
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder

    def predict(
            self,
            image: np.ndarray,
            teacher_mean: np.ndarray,
            teacher_std: np.ndarray,
            q_st_start: Optional[np.ndarray] = None,
            q_st_end: Optional[np.ndarray] = None,
            q_ae_start: Optional[np.ndarray] = None,
            q_ae_end: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        EXACT logic as efficientAD.py (your provided source):
          teacher_out = (teacher - mean) / std
          map_st = mean((teacher_out - student[:384])^2)
          map_ae = mean((ae_out - student[384:])^2)
          quantile scaling with 0.1 multiplier
          map_combined = 0.5*st + 0.5*ae
        """
        teacher_output = self.teacher.run_sync(image).astype(np.float32)
        student_output = self.student.run_sync(image).astype(np.float32)
        autoencoder_output = self.autoencoder.run_sync(image).astype(np.float32)

        # teacher normalization (std is variance-like per your source)
        teacher_output = (teacher_output - teacher_mean) / teacher_std

        # fixed split confirmed by your shapes: 384 + 384 = 768
        map_st = np.mean((teacher_output - student_output[:, :, :384]) ** 2, axis=2, keepdims=True)
        map_ae = np.mean((autoencoder_output - student_output[:, :, 384:]) ** 2, axis=2, keepdims=True)

        # quantile scaling (source)
        if q_st_start is not None:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
        if q_ae_start is not None:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

        map_combined = 0.5 * map_st + 0.5 * map_ae
        return map_combined, map_st, map_ae

    @staticmethod
    def teacher_normalization(teacher, train_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        EXACT efficientAD.py teacher_normalization().

        NOTE: teacher_std here is variance-like (mean squared deviation), matching your source.
        """
        output_means = []
        for images, _, _ in train_loader:
            output = teacher.run_sync(images[0]).astype(np.float32)
            output_means.append(np.mean(output, axis=(0, 1)))

        channel_mean = np.mean(np.stack(output_means, axis=0), axis=0)[None, None, :]

        output_stds = []
        for images, _, _ in train_loader:
            output = teacher.run_sync(images[0]).astype(np.float32)
            distance = (output - channel_mean) ** 2
            output_stds.append(np.mean(distance, axis=(0, 1)))

        channel_std = np.mean(np.stack(output_stds, axis=0), axis=0)[None, None, :]
        return channel_mean, channel_std

    def map_normalization(
            self,
            teacher_mean: np.ndarray,
            teacher_std: np.ndarray,
            loader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        EXACT efficientAD.py map_normalization():
          quantiles on raw maps (no quantile scaling while collecting)
        """
        maps_st = []
        maps_ae = []

        for images, _, _ in loader:
            _, map_st, map_ae = self.predict(
                image=images[0],
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                q_st_start=None, q_st_end=None,
                q_ae_start=None, q_ae_end=None
            )
            maps_st.append(map_st)
            maps_ae.append(map_ae)

        maps_st = np.concatenate(maps_st)
        maps_ae = np.concatenate(maps_ae)

        q_st_start = np.quantile(maps_st, 0.9)
        q_st_end   = np.quantile(maps_st, 0.995)
        q_ae_start = np.quantile(maps_ae, 0.9)
        q_ae_end   = np.quantile(maps_ae, 0.995)

        return q_st_start, q_st_end, q_ae_start, q_ae_end
