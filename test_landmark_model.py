import cv2
import numpy as np
from collections import deque

from detector.hand_detector import Hand_Detector
from data.loaders import Landmark_Loader
from models import landmark_model
from models.landmark_model import Landmark_Model
from video.camera import Video

# my saved weights
WEIGHTS_PATH = "landmark_pretrained.pt"

# predictions seem to fluctuate a lot if i predict on every frame
# so i leaen the lesson from sensors section and add a moving average
# it now takes 10 frames, averages them and predicts on the average
WINDOW_SIZE = 10

# the threshhold to bellow it would be uncertain (i draw it as "?" because the word is too long for my camera feed)
MIN_CONFIDENCE = 0.50


def build_model():
    # i only call the loader to get the class names from it
    # in heindsight i should have probaly make the model store the names nativly somewhere
    loader = Landmark_Loader()
    loader.get_paths_and_labels()
    class_names = loader.class_names

    # i make my model and load the weights
    model = Landmark_Model(
        normalize=True,
        loaders=None,
        num_classes=len(class_names),
        class_names=class_names,
        best_model_path=WEIGHTS_PATH,
    )
    model.load_model()

    # set the model to eval
    model.model.eval()

    return model


def main():
    model = build_model()
    video = Video()
    hand_detector = Hand_Detector()

    # use a deque to manage the sliding average
    prediction_buffer = deque(maxlen=WINDOW_SIZE)

    # claude recommened putting the testing loop inside a try finaly block
    # that way if any thing fails it closes every thing

    try:
        while True:
            frame = video.get_frame()

            if frame is None:
                print("cant read frame")
                break

            # remember that the traing data was left handed so i need to flip the frame
            # before feeding it to the model, 1 is for a horizontal flip , 0 for vertical
            frame = cv2.flip(frame, 1)

            # detect hands
            landmarks = hand_detector.get_landmarks(frame)

            # if there is no hands in the frame clear he deque and draw a message
            if landmarks is None:
                prediction_buffer.clear()
                video.draw_text(frame, "No hand detected")

            else:
                video.draw_landmark_points(frame, landmarks)

                # get the probability distrobution, and add it to the deque
                probabilities = model.predict_probabilities(landmarks)
                prediction_buffer.append(probabilities)

                # dont make predictions if the deque is not full yet
                if len(prediction_buffer) < WINDOW_SIZE:
                    # Wait until the window is full before showing a label.
                    video.draw_text(
                        frame, f"Stabilizing: {len(prediction_buffer)}/{WINDOW_SIZE}"
                    )

                else:
                    # average all the probabilities in the deque
                    average_probabilities = np.mean(np.stack(prediction_buffer), axis=0)

                    # get the index of the letter with the highest probability
                    predicted_index = int(np.argmax(average_probabilities))
                    # get the letter with that index
                    predicted_letter = model.class_names[predicted_index]
                    # get the probility of that letter from the probility distrobution
                    confidence = float(average_probabilities[predicted_index])

                    # i also save the current frames predection to compare it with the averaged one
                    raw_letter = model.class_names[int(np.argmax(probabilities))]

                    # i show low confidence as  as "?" because writign it is too long
                    if confidence >= MIN_CONFIDENCE:
                        label = f"{predicted_letter} ({confidence:.0%})"
                    else:
                        label = f"? ({confidence:.0%})"

                    # i finaly drow the lable and probabily on the screen with the current frames predection
                    video.draw_text(frame, f"{label}  now: {raw_letter}")

            # actualy draw the frame in a window
            # display_frame returns false once 'q' is pressed if tht happens it breaks the loop
            if not video.display_frame(frame):
                break

    finally:
        # Video has no close() method, so the camera is released directly.
        video.cap.release()
        hand_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
