import cv2
import numpy as np
import mediapipe as mp


# for the cnn model i cant just feed it the full webcame feed
# i need to crop it to only the hand to be simmiler to what the model trained on
# i already explained model parameters in the repors so i wont here,
class Hand_Detector:
    def __init__(
        self,
        max_hands=1,
        # this is the 4th time i tweek those values btw, this time it seems to work ok
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        padding=30,
    ):
        # i crop the box slighlty larger than the hand to match the traingin data
        self.box_padding = padding
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_box(self, frame):
        height, width, _ = frame.shape

        # claude recommended cvtColor this method when reviewing, it works like torch premute
        # but spicicly for images (here it changes cv2 blue green red capture to reg green blue)
        # which is what mediapipe hands expect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # run the frame through the hand solution
        result = self.hands.process(rgb)

        # as i discused in the report media pipe hands returns 21 3d points for spicific hand landmarks
        # to get the box to crop the image at, all i need to do i just get the min and max coordinates
        # from these landmarks in 2d
        # i only work with one hand for now, but claude recommended to loop through all avilable hands in case i wont
        # to add multi hand support in the future
        boxes = []
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                # get all xs and ys in all landmark points,
                # also these points return as normailized values from 0 to 1 so
                # to make it actually detect where the hand on the feed i need to multiply by the width and height
                # the int casting is because there are pixels, and there is no half a pixel XD
                all_x = [int(point.x * width) for point in hand.landmark]
                all_y = [int(point.y * height) for point in hand.landmark]

                # get the min and max x and y, i know this looks complicated but is really now
                # you get the min/max point and subsract/add the padding value to create the padding box
                # the outer min, max are just so that the box coordinates never go out of bounds
                x1 = max(0, int(min(all_x)) - self.box_padding)
                y1 = max(0, int(min(all_y)) - self.box_padding)
                x2 = min(width, int(max(all_x)) + self.box_padding)
                y2 = min(height, int(max(all_y)) + self.box_padding)

                boxes.append((x1, y1, x2, y2))

        return boxes

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        landmarks = result.multi_hand_landmarks[0].landmark

        # get the 21 3d points and return them as a numpy array
        points = np.zeros((21, 3))
        for i, landmark in enumerate(landmarks):
            points[i, 0] = landmark.x
            points[i, 1] = landmark.y
            points[i, 2] = landmark.z

        return points.flatten()

    def crop_hand(self, frame, box):
        x1, y1, x2, y2 = box
        # look at this buetiful use of slicing
        return frame[y1:y2, x1:x2]

    def close(self):
        self.hands.close()
