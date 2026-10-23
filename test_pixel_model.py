import cv2

from detector.hand_detector import Hand_Detector
from data.loaders import Pixel_Loader
from models.pixel_models import CNN_Model
from video.camera import Video

# the saved weights
WEIGHTS_PATH = "pixel_pretrained.pt"

# predictions below this probability are shown as "?" in the label
MIN_CONFIDENCE = 0.5


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

    # Video raises its own error if the camera cannot be opened, so no manual
    # check is needed here. It is created before the detector so a camera
    # failure leaves nothing to clean up.
    video = Video()
    detector = Hand_Detector(max_hands=1, static_image_mode=False)

    # claude recommended the try finaly block to garantee that the detecor is close
    try:
        while True:
            frame = video.get_frame()

            # None means the camera stopped delivering frames
            if frame is None:
                break

            # detect the hand
            boxes = detector.detect_box(frame)

            if not boxes:
                video.draw_text(frame, "No hand detected")

            for box in boxes:
                # slice the hand box from the frame to feed it to the model
                crop = detector.crop_hand(frame, box)

                # if there is no hand, skip prediction in this frame
                if crop.size == 0:
                    continue

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                # use my classes to get predectit class names and probability
                name, probability = model.predict(rgb_crop)

                # Drawing happens after predict() because the crop shares memory
                # with the frame, so drawing earlier would alter the model's input.
                video.draw_box(frame, box)

                # Low confidence is shown as "?", the same as in the landmark script.
                if probability >= MIN_CONFIDENCE:
                    label = f"{name} ({probability:.0%})"
                else:
                    label = f"? ({probability:.0%})"

                video.draw_text(frame, label)

            # display_frame returns False once 'q' is pressed, which ends the loop
            if not video.display_frame(frame):
                break
    finally:
        # close MediaPipe, the camera and the windows even if an error occurs.
        # Video has no close() method, so the camera is released directly.
        detector.close()
        video.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
