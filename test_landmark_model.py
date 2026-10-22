import cv2
import numpy as np
from detector.hand_detector import Hand_Detector
from data.loaders import Landmark_Loader
from models.landmark_model import Landmark_Model


# Saved landmark model weights
WEIGHTS_PATH = "landmark_pretrained.pt"

# Draw a red label when confidence is below this threshold
MIN_CONFIDENCE = 0.5

# BGR colors
COLOR_CONFIDENT = (0, 255, 0)
COLOR_UNCERTAIN = (0, 0, 255)


def build_model():
    # Get the same class-name mapping used during training
    loader = Landmark_Loader(normalize=True)
    loader.get_paths_and_labels()
    class_names = loader.class_names

    model = Landmark_Model(
        loaders=None,
        num_classes=len(class_names),
        class_names=class_names,
        normalize=True,
    )

    model.load_model()
    return model


def main():
    model = build_model()
    detector = Hand_Detector(max_hands=1)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Can't open camera")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            # Extract the 21 landmarks as a flattened (63,) array.
            landmarks = detector.get_landmarks(frame)

            if landmarks is None:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_UNCERTAIN,
                    2,
                )
            else:
                # predict() handles reshaping and normalization.

                # Original orientation
                name1, prob1 = model.predict(landmarks)

                # Mirrored orientation: negate x coordinates
                mirrored = landmarks.reshape(21, 3).copy()
                mirrored[:, 0] *= -1

                name2, prob2 = model.predict(mirrored)

                # Choose the prediction with the higher confidence
                if prob1 >= prob2:
                    name, probability = name1, prob1
                else:
                    name, probability = name2, prob2
                color = (
                    COLOR_CONFIDENT
                    if probability >= MIN_CONFIDENCE
                    else COLOR_UNCERTAIN
                )

                cv2.putText(
                    frame,
                    f"{name} {probability:.0%}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )

                # Draw the 21 detected landmarks.
                height, width = frame.shape[:2]
                points = landmarks.reshape(21, 3)

                for x, y, _ in points:
                    px = int(x * width)
                    py = int(y * height)

                    # Only draw points that fall inside the frame.
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(
                            frame,
                            (px, py),
                            4,
                            (255, 200, 0),
                            -1,
                        )

            cv2.imshow("Landmark Model", frame)

            # Press Q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
