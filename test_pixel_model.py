import cv2

from detector.hand_detector import Hand_Detector
from data.loaders import Pixel_Loader
from models.pixel_models import CNN_Model

# the saved weights
WEIGHTS_PATH = "pixel_pretrained.pt"

# i use this threshhold to draw a redbox around low proba predections
MIN_CONFIDENCE = 0.5

# these colors are in BGR not rgb
COLOR_CONFIDENT = (0, 255, 0)
COLOR_UNCERTAIN = (0, 0, 255)


def build_model():
    loader = Pixel_Loader()

    loader.get_paths_and_labels()
    class_names = loader.class_names

    model = CNN_Model(
        loaders=None,
        num_classes=len(class_names),
        class_names=class_names,
        image_size=loader.image_size,
        grey_scale=loader.grey_scale,
    )

    model.load_model(WEIGHTS_PATH)
    return model


def main():
    model = build_model()

    detector = Hand_Detector(max_hands=1)

    # open webcam 0 as a stream
    cap = cv2.VideoCapture(0)
    # cap.isOpened is just a flag to check if the camera is open
    # if it doesnt i just reaise an error and let the program crash
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"cant open camera")

    # claude recommended the try finaly block to garantee that the detecor is close
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # detect the hand
            boxes = detector.detect_box(frame)

            # the drawing on the frame is mostly just generated to be honest
            if not boxes:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_UNCERTAIN,
                    2,
                )

            for box in boxes:
                # slice the hand box from the frame to feed it to the model
                crop = detector.crop_hand(frame, box)

                # if there is no hand, skip prediction in this frame
                if crop.size == 0:
                    continue

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                # use my classes to get predectit class names and probability
                name, probability = model.predict(rgb_crop)

                # drawing the text on the frame, i kinda understand how this works
                # but it is also just generated with copilot
                color = (
                    COLOR_CONFIDENT
                    if probability >= MIN_CONFIDENCE
                    else COLOR_UNCERTAIN
                )
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{name} {probability:.0%}",
                    # max() keeps the label on-screen when the box touches the top edge.
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

            # draw the frame
            cv2.imshow("frame", frame)

            # cv2.waitKey waits for a key press to close the window in this case it is "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # close MediaPipe, the camera and the windows even if an error occurs.
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
