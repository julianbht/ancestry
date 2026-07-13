"""Turn a detected face box into a crop. Pure image geometry — no I/O, no state."""

import numpy as np

from pipeline.face_crop.detector import FaceDetection


def crop_face(img: np.ndarray, face: FaceDetection, margin_frac: float) -> np.ndarray:
    """Crop the face box from img, padded by margin_frac of the box size on each
    side and clamped to the frame bounds.

    Raises ValueError if the (clamped) box is empty.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = face.box_xyxy

    mx = (x1 - x0) * margin_frac
    my = (y1 - y0) * margin_frac

    cx0 = max(0, int(round(x0 - mx)))
    cy0 = max(0, int(round(y0 - my)))
    cx1 = min(w, int(round(x1 + mx)))
    cy1 = min(h, int(round(y1 + my)))

    if cx1 <= cx0 or cy1 <= cy0:
        raise ValueError(f"Empty face crop after clamping: box={face.box_xyxy}")

    return img[cy0:cy1, cx0:cx1]
