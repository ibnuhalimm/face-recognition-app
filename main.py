import tkinter as tk
import tkinter.ttk as ttk
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
from recognition import Recognition
from devices import Devices


class Main:
    def __init__(self, window, webcam_sources):
        self.window = window
        self.webcam_sources = webcam_sources
        self.webcam_names = self.webcam_names(self.webcam_sources)

        self.face_image_name = 'face.jpg'
        self.face_image = os.path.join(os.getcwd(), self.face_image_name)
        self.delete_face_image()

        self.photo_taken = False

        self.video_source = self.webcam_sources[0]
        self.video_capture(self.video_source)
        self.make_ui()

        self.stream_video()

    def delete_face_image(self):
        if os.path.exists(self.face_image):
            os.remove(self.face_image)

    def webcam_names(self, webcam_sources):
        names = {}
        i = 1
        for x in webcam_sources:
            names["Camera " + str(i)] = x
            i += 1
        return names

    def change_webcam(self, event):
        self.video_source = self.get_webcam_index(self.camera_option.get())

        self.delete_face_image()
        self.text_label.config(background="yellow",
                               foreground="black",
                               text="Ready")

        self.vid_cap.release()
        self.video_capture(self.video_source)

    def get_webcam_index(self, selected_name):
        for name, cam_index in self.webcam_names.items():
            if name == selected_name:
                return cam_index;
        return 0

    def get_webcam_name(self, selected_index):
        for name, cam_index in self.webcam_names.items():
            if cam_index == selected_index:
                return name
        return "Camera 1"

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

        self.window_style = ttk.Style()
        self.window_style.theme_use('clam')
        self.window_style.configure("TCombobox")

        self.main_frame = tk.Frame(master=self.window)
        self.main_frame.pack()

        self.label_name_frame = tk.Frame(master=self.window,
                                         width=self.vid_width,
                                         padx=10,
                                         pady=10,
                                         bg="white")
        self.label_name_frame.pack(fill=tk.BOTH,
                                   expand=True)

        self.toolbar_button_frame = tk.Frame(master=self.window,
                                             width=self.vid_width,
                                             padx=10,
                                             pady=10,
                                             bg="white")
        self.toolbar_button_frame.pack(fill=tk.BOTH,
                                       expand=True)

        self.canvas = tk.Canvas(master=self.main_frame,
                                width=self.vid_width,
                                height=self.vid_heigth)
        self.canvas.pack()

        self.canvas_photo = tk.Canvas(master=self.main_frame,
                                      width=self.vid_width,
                                      height=self.vid_heigth)

        self.text_label = ttk.Label(master=self.toolbar_button_frame)
        self.text_label.config(background="yellow",
                               text="Ready",
                               width=25,
                               font=(None, 12, "bold"),
                               anchor="center")
        # self.text_label.pack(fill=tk.BOTH,
        #                      expand=True)
        self.text_label.grid(row=1,
                             column=1,
                             padx=1,
                             ipadx=5,
                             ipady=5)

        self.camera_var = tk.StringVar()
        self.camera_var.set(self.get_webcam_name(self.video_source))
        self.camera_option = ttk.Combobox(self.toolbar_button_frame,
                                          textvariable=self.camera_var,
                                          values=list(self.webcam_names.keys()),
                                          state="readonly",
                                          width=15)
        self.camera_option.bind('<<ComboboxSelected>>', self.change_webcam)
        self.camera_option.grid(row=1,
                                column=2,
                                padx=5,
                                ipadx=5,
                                ipady=5)

        self.button_new = ttk.Button(master=self.toolbar_button_frame,
                                     text="New Window",
                                     command=self.new_window)
        self.button_new.grid(row=1,
                             column=3,
                             padx=5,
                             ipady=1)

        # self.button_take_photo = ttk.Button(master=self.toolbar_button_frame,
        #                                     text="Take Photo",
        #                                     command=self.take_photo_manual)
        # self.button_take_photo.grid(row=1,
        #                             column=4,
        #                             padx=5,
        #                             ipady=1)

    def take_photo_manual(self):
        is_reading, frame = self.vid_cap.read()

        if is_reading:
            self.take_photo(frame=frame,
                            is_manual=True)

    def stream_video(self):
        frontal_face = os.path.join(os.getcwd(), 'frontalface.xml')
        face_cascade = cv2.CascadeClassifier(frontal_face)
        is_reading, frame = self.vid_cap.read()

        if is_reading:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img,
                                                  scaleFactor=1.25,
                                                  minNeighbors=2,
                                                  minSize=(96, 96),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_crop_color = frame[y:y+h, x:x+w]
                face_crop_gray = gray_img[y:y+h, x:x+w]

                blur_threshold = cv2.Laplacian(face_crop_gray,
                                               cv2.CV_64F).var()

                self.text_label.config(background="yellow",
                                       text="Detecting Face...")
                self.text_label.config(background="yellow",
                                       text=blur_threshold)

                if blur_threshold >= 50:
                    self.take_photo(face_crop_color)

            flipped_img = cv2.flip(frame, 1)
            flipped_img = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB)

            img_array = Image.fromarray(flipped_img)
            self.photo_image = ImageTk.PhotoImage(image=img_array)
            self.canvas.create_image(0,
                                     0,
                                     image=self.photo_image,
                                     anchor=tk.NW)

        self.canvas.after(10, self.stream_video)

    def take_photo(self, frame, is_manual = False):
        if is_manual:
            now = datetime.now()
            timetamp = now.strftime('%Y%M%d%H%m%s')
            photos_folder = os.path.join(os.getcwd(), 'photos')
            photo_filename = f'{photos_folder}/photo_{timetamp}.jpg'

            self.flipped_frame = cv2.flip(frame, 1)
            cv2.imwrite(photo_filename, self.flipped_frame)
            return

        if not os.path.exists(self.face_image) and not is_manual:
            self.canvas.after_cancel(self.stream_video)
            self.vid_cap.release()

            self.text_label.config(background="green",
                                   foreground="white",
                                   text="Photo Taken")

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

            self.recognition = Recognition()
            self.recognition.set_photo_image(self.face_image_name)
            self.text_label.config(text=self.recognition.get_name())

    def get_face_image(self):
        return self.face_image

    def close_window(self):
        self.vid_cap.release()
        self.window.destroy()

    def new_window(self):
        self.close_window()
        main()


def main():
    devices = Devices()
    cam_devices = devices.list_ports()

    app = tk.Tk()
    Main(app, cam_devices)
    app.mainloop()


if __name__ == '__main__':
    main()
