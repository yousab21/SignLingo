from data.loaders import Pixel_Loader, Landmark_Loader
from models.pixel_models import CNN_Model
from models.landmark_model import Landmark_Model
from video.camera import Video
from detector.hand_detector import Hand_Detector
from test_landmark_model import Landmark_Model_Tester
from test_pixel_model import pixel_model_tester
import os
import cv2
import numpy as np
from collections import deque

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PIXEL_WEIGHTS_PATH = "pixel_pretrained.pt"
LANDMARK_WEIGHTS_PATH = "landmark_pretrained.pt"


def clear_terminal():
    os.system("clear")


def load_pixel_model():
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

    model.load_model(PIXEL_WEIGHTS_PATH)
    return model


def load_landmark_model():
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
        best_model_path=LANDMARK_WEIGHTS_PATH,
    )
    model.load_model()

    # set the model to eval
    model.model.eval()

    return model


def model_selection():
    clear_terminal()
    print(f"=" * 20)
    print(f"signLingo")
    print(f"=" * 20)

    print()
    print(f"chose your model")
    print(f"-" * 20)
    print(f"1 - pixel model")
    print(f"2 - landmark model")
    print(f"-" * 20)
    while True:
        choice = input("choice: ")
        if choice == "1":
            return load_pixel_model()
        elif choice == "2":
            return load_landmark_model()
        else:
            print("invalid choice")


def get_user_word():
    clear_terminal()
    word = input("type a word you would like to learn to hand spell: ")
    print()
    return word


def show_guide_image(letter):
    path = BASE_DIR / "data" / "guide-images" / f"{letter}.png"
    Video().draw_image(str(path), letter.upper())


# if you thought that sequintial code is ugly brace yourself becuase it is
# about to get REAL ugly with this next function
def sign_char(model, char, window_size=10, right_passes=3):
    char = char.lower()

    # check if the model is a landmark model else it is the piel model
    is_landmark = model.name == "Landmark_Model"
    is_pixel = not is_landmark
    # landmark model classes are in capital but pixel classesa are in small
    # this just cahnges the char from the functio to the correct case
    char = char.upper() if is_landmark else char.lower()

    # if the char is not in the model clases raise an ValueError
    # i cant hard code it being a letter beacuse the pixel model is also trained on numbers\
    if char not in model.class_names:
        raise ValueError(f"Unknown character: {char}")

    # prepare the my objects
    video = Video()
    detector = Hand_Detector()

    # make the deque for the window averaging
    buffer = deque(maxlen=window_size)

    # i pretty much copied that from the tester script and just changes some var names
    # i wont rexplain it, but i will just explain the new logic
    try:
        count_right = 0
        while True:
            frame = video.get_frame()
            predicted_char = None
            confidence = 0.0

            if frame is None:
                raise RuntimeError("camera is not working")

            if is_landmark:
                frame = cv2.flip(frame, 1)
                landmarks = detector.get_landmarks(frame)

                if landmarks is None:
                    buffer.clear()
                    video.draw_text(frame, "No hand detected")

                else:
                    video.draw_landmark_points(frame, landmarks)
                    probabilities = np.asarray(
                        model.predict_probabilities(landmarks)
                    ).reshape(-1)
                    buffer.append(probabilities)

                    if len(buffer) == window_size:
                        average = np.mean(np.stack(buffer), axis=0)

                        index = int(np.argmax(average))
                        predicted_char = model.class_names[index]
                        confidence = float(average[index])

                    else:
                        predicted_char = None
                        confidence = 0.0
            # ============================================================
            # pixel model section

            if is_pixel:
                boxes = detector.detect_box(frame)

                if not boxes:
                    buffer.clear()
                    confidence = 0
                    predicted_char = None
                    video.draw_text(frame, "No hand detected")

                for box in boxes:
                    crop = detector.crop_hand(frame, box)

                    if crop.size == 0:
                        continue

                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                    predicted_char, probability = model.predict(rgb_crop)

                    buffer.append((predicted_char, float(probability)))

                    # remember that i didnt implenent a probability distribution for the pixel moel
                    # so i would just haev to use the frame probability as the final one
                    confidence = probability
                    video.draw_box(frame, box)

            # ==========================================
            #  by now we should have a predicted char and a confidence regardless of the model
            # if the model sees a hand other wise it would be none and 0
            if predicted_char is not None and confidence > 0:
                if predicted_char == char:
                    count_right += 1
                    video.draw_text(frame, "Good!")
                else:
                    count_right = 0
                    video.draw_text(frame, f"detectign [{predicted_char}], Try again")

                if count_right >= right_passes:
                    video.display_frame(frame)
                    return True

            # i already have a way for the user to close the window with q so
            # i extend ites use to make it skip the letter if q was pressed
            # as my model has some letter it just refuse to get right
            if not video.display_frame(frame):
                return False
    finally:
        video.cap.release()
        detector.close()
        cv2.destroyAllWindows()


def model_menu(model):
    clear_terminal()
    print()
    print(f"chose operation")
    print(f"-" * 20)
    print(f"1 - test model")
    print(f"2 - run app")
    print(f"-" * 20)

    while True:
        choice = input("choice: ")
        if choice == "1":
            if model.name == "Landmark_Model":
                tester = Landmark_Model_Tester()
            else:
                tester = pixel_model_tester()

            tester.run_test()
            break

        elif choice == "2":
            clear_terminal()
            word = get_user_word()
            for letter in word:
                show_guide_image(letter)
                correct = sign_char(model, letter)

            clear_terminal()
            print("-" * 20)
            print(
                "now that you know the signs try signing them without the refrence images"
            )
            print(
                f"you will need to sign the letters for the word [{word}] one after the other "
            )
            print("-" * 20)
            print("press enter to start")
            input()
            for letter in word:
                correct = sign_char(model, letter)
            break

        else:
            print("invalid choice")


def main():
    model = model_selection()
    model_menu(model)


if __name__ == "__main__":
    main()
