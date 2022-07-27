import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os


class Main:
    def __init__(self, window):
        self.window = window

        self.face_image_name = 'face.jpg'
        self.face_image = os.path.join(os.getcwd(), self.face_image_name)
        self.delete_face_image()

        self.photo_taken = False

        self.video_source = 2
        self.video_capture(self.video_source)
        self.make_ui()

        self.stream_video()

    def delete_face_image(self):
        if os.path.exists(self.face_image):
            os.remove(self.face_image)

    def video_capture(self, video_source=0):
        self.vid_cap = cv2.VideoCapture(video_source)
        self.vid_width = 640
        self.vid_heigth = 480

        if not self.vid_cap.isOpened():
            raise Exception("Could not open video source", video_source)

        self.vid_cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.vid_cap.set(3, self.vid_width)
        self.vid_cap.set(4, self.vid_heigth)

    def make_ui(self):
        self.window['bg'] = 'white'
        self.window.title("Face Masked Recognition with CNN")
        self.window.resizable(width=False,
                              height=False)

        self.main_frame = tk.Frame(master=self.window)
        self.main_frame.pack()

        self.bottom_frame = tk.Frame(master=self.window,
                                     width=self.vid_width,
                                     padx=10,
                                     pady=10,
                                     bg="white")
        self.bottom_frame.pack(fill=tk.BOTH,
                               expand=True)

        self.canvas = tk.Canvas(master=self.main_frame,
                                width=self.vid_width,
                                height=self.vid_heigth)
        self.canvas.pack()

        self.canvas_photo = tk.Canvas(master=self.main_frame,
                                      width=self.vid_width,
                                      height=self.vid_heigth)

        self.button_new = tk.Button(master=self.bottom_frame,
                                    text="NEW CAMERA",
                                    command=self.new_window)
        self.button_new.pack(side=tk.LEFT)

        self.text_label = tk.Label(master=self.bottom_frame,
                                   text="Ready",
                                   bg="yellow",
                                   fg="black",
                                   padx=5,
                                   pady=5,
                                   width=25)
        # self.text_label.config(font=(None, 10))
        self.text_label.pack(side=tk.RIGHT,
                             padx=10)

    def stream_video(self):
        frontal_face = os.path.join(os.getcwd(), 'frontalface.xml')
        face_cascade = cv2.CascadeClassifier(frontal_face)
        ret, frame = self.vid_cap.read()

        if ret:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img,
                                                  scaleFactor=1.05,
                                                  minNeighbors=5,
                                                  minSize=(100, 100),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_crop_color = frame[y:y+h, x:x+w]
                face_crop_gray = gray_img[y:y+h, x:x+w]

                blur_threshold = cv2.Laplacian(face_crop_gray,
                                               cv2.CV_64F).var()

                self.text_label['text'] = 'Detecting Face...'

                if blur_threshold >= 85:
                    self.take_photo(face_crop_color)

            flipped_img = cv2.flip(frame, 1)
            flipped_img = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)

            img_array = Image.fromarray(flipped_img)
            self.photo_image = ImageTk.PhotoImage(image=img_array)
            self.canvas.create_image(0,
                                     0,
                                     image=self.photo_image,
                                     anchor=tk.NW)

        self.window.after(10, self.stream_video)

    def take_photo(self, frame):
        if not os.path.exists(self.face_image):
            self.window.after_cancel(self.stream_video)
            self.vid_cap.release()

            self.text_label['bg'] = 'green'
            self.text_label['fg'] = 'white'
            self.text_label['text'] = 'Photo Taken'

            self.flipped_frame = cv2.flip(frame, 1)
            cv2.imwrite(self.face_image_name, self.flipped_frame)

            img_array = Image.fromarray(self.flipped_frame)
            photo_image = ImageTk.PhotoImage(image=img_array)

            self.canvas.delete('all')

            self.canvas.pack()
            self.canvas.create_image(0,
                                     0,
                                     image=photo_image,
                                     anchor=tk.NW)

    def get_face_image(self):
        return self.face_image

    def close_window(self):
        self.vid_cap.release()
        self.window.destroy()

    def new_window(self):
        self.close_window()
        main()


def main():
    app = tk.Tk()
    Main(app)
    app.mainloop()


if __name__ == '__main__':
    main()
