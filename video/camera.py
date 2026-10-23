import cv2


class Video:
    def __init__(
        self,
        drawing_color=(0, 255, 0),  # green in BGR
    ):
        self.drawing_color = drawing_color
        # this returns the camera feed at index 0, not the actual frame (ask me how i know XD)
        # to get the actual frame i need to call read() on it`]`
        self.cap = cv2.VideoCapture(0)
        # claude recommende raising an error if the camera is not working
        if not self.cap.isOpened():
            raise RuntimeError("camera is not working")

    def get_frame(self):
        read_ok, frame = self.cap.read()
        # read ok is just a flag to check if the frame was read
        if not read_ok:
            return None

        return frame

    def display_frame(self, frame):
        # imshow() hands the frame window to my window manager (hyprland)
        # but it doesnt refresh it or adds a way to close it so it just freezes the window
        cv2.imshow("frame", frame)
        # that is were this cv2.waitKey(delay) comes in
        # it does 2 things, it refreshes the window so the new frame acrually renders
        # and it also adds a delay (here it is 1ms) in which it tries to read user input, as the delay is so small
        # it is not noticable from the video feed, but it adds a way to read user input without freezeing the window
        key = cv2.waitKey(1)

        # the returned key is in ascii , so to compare code, so to be able to comapre it with actual chars
        # we need to do some bit wise shananigs
        # key & 0XFF is a bitwise and with 11111111 which should give the exact same value as the key
        # exepct the 8 least significant bits which are the ascii char code, as cv also encodes some stuff after those bits that i dont care about
        # ord(char) just gets the ascii code of the char
        # so key & 0XFF == ord(q) just checks if the user pressed q, for which i make the method return false and close the window
        if (key & 0xFF) == ord("q"):
            return False

        return True

    def draw_box(self, frame, box):
        x1, y1, x2, y2 = box
        # this 2 is just the thickness of the lines
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.drawing_color, 2)

    def draw_text(self, frame, text):
        # iam not gonna expain every parameter here is the funtion deffinition from the docs
        # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
        # orh is the bottom left corner of the text box, wich is why i dont start from 0,0 for the top corner
        cv2.putText(
            frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.drawing_color, 2
        )

    def draw_landmark_points(self, frame, points):
        height, width, _ = frame.shape

        for x, y, _ in points.reshape(21, 3):
            # remember thay the points are normalized so from 0 to 1
            # so to redraw them i need to multiply them by the frame width and height
            px = int(x * width)
            py = int(y * height)

            # this ignore points outside the frame

            if 0 <= px < width and 0 <= py < height:
                cv2.circle(
                    frame,
                    (px, py),
                    radius=5,
                    color=self.drawing_color,
                    thickness=-1,
                )

    def draw_image(self, path, letter=None):
        image = cv2.imread(path)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        self.draw_text(image, f"Sign for letter [ {letter} ]")
        cv2.imshow("image", image)
        # wait key 0 indefinitely for a key to be pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
