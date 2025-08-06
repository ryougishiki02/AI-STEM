import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")
        self.root.geometry("1400x1000")
        self.root.resizable(True, True)

        self.video_path = ""
        self.cap = None
        self.playing = False
        self.paused = False
        self.start_frame = None
        self.end_frame = None
        self.roi = None

        # GUI Elements
        self.canvas_width = 640
        self.canvas_height = 480

        # Top Frame for video and ROI display
        self.top_frame = tk.Frame(root)
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        self.video_canvas = tk.Canvas(self.top_frame, bg="white")
        self.video_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.roi_canvas = tk.Canvas(self.top_frame, bg="white")
        self.roi_canvas.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Bottom Frame for start and end frame display
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.start_frame_canvas = tk.Canvas(self.bottom_frame, bg="white")
        self.start_frame_canvas.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.end_frame_canvas = tk.Canvas(self.bottom_frame, bg="white")
        self.end_frame_canvas.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Control Frame for buttons and progress bar
        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.open_button = tk.Button(self.control_frame, text="Open Video", command=self.open_video)
        self.open_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.play_button = tk.Button(self.control_frame, text="Play", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.pause_button = tk.Button(self.control_frame, text="Pause", command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.progress = tk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        self.options_frame = tk.Frame(root)
        self.options_frame.grid(row=3, column=0, columnspan=2, sticky="ew")

        self.roi_button = tk.Button(self.options_frame, text="Select ROI", command=self.select_roi)
        self.roi_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.start_frame_button = tk.Button(self.options_frame, text="Set Start Frame", command=self.set_start_frame)
        self.start_frame_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.end_frame_button = tk.Button(self.options_frame, text="Set End Frame", command=self.set_end_frame)
        self.end_frame_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = tk.Button(self.options_frame, text="Save Video", command=self.save_video)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.progress_label = tk.Label(root, text="")
        self.progress_label.grid(row=4, column=0, columnspan=2)

        self.video_img_label = None
        self.roi_img_label = None
        self.start_img_label = None
        self.end_img_label = None

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.top_frame.grid_rowconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.wmv")])
        if not self.video_path:
            return
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress.config(to=self.total_frames - 1)

    def play_video(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Please open a video file first.")
            return
        self.playing = True
        self.paused = False
        self.update_frame()

    def pause_video(self):
        self.playing = False
        self.paused = True
        self.update_frame()

    def seek(self, frame_number):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = self.cap.read()
            if ret:
                self.show_frame(frame)
            else:
                messagebox.showerror("Error", "Failed to read frame.")

    def select_roi(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Please open a video file first.")
            return
        self.playing = False
        self.paused = True

        # 读取当前帧（保持原始尺寸）
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read frame.")
            return

        # 获取原始视频帧的尺寸
        self.original_frame = frame
        self.original_height, self.original_width = frame.shape[:2]

        # 将帧缩放到 800x800 的显示尺寸（新代码）
        frame_resized = self.resize_image_to_fit_canvas(frame, 800, 800)
        self.resized_height, self.resized_width = frame_resized.shape[:2]

        #  计算缩放比例（新代码）
        self.scale_x = self.original_width / self.resized_width
        self.scale_y = self.original_height / self.resized_height

        self.max_scale = max(self.scale_x, self.scale_y) #1
        print(self.original_width, self.original_height, self.resized_width, self.resized_height, self.scale_x, self.scale_y)

        #  创建固定大小为 1024x1024 的 ROI 选择窗口（新代码）
        self.roi_window = tk.Toplevel(self.root)
        self.roi_window.title("Select ROI")
        self.roi_window.geometry("1024x1024")

        self.roi_selection_canvas = tk.Canvas(self.roi_window, width=800, height=800)
        self.roi_selection_canvas.pack()

        # 6️ 显示缩放后的图像（保持比例）
        self.roi_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
        self.roi_selection_canvas.create_image(0, 0, anchor=tk.NW, image=self.roi_img)

        # 7 绑定鼠标事件，捕获用户的 ROI 选择
        self.roi_selection_canvas.bind("<ButtonPress-1>", self.start_roi)
        self.roi_selection_canvas.bind("<B1-Motion>", self.draw_roi)
        self.roi_selection_canvas.bind("<ButtonRelease-1>", self.end_roi)

        self.roi_start = None
        self.roi_rect = None

        # 确认选择按钮
        confirm_button = tk.Button(self.roi_window, text="Confirm", command=self.confirm_roi)
        confirm_button.pack(side=tk.RIGHT, padx=10)

    def start_roi(self, event):
        self.roi_start = (event.x, event.y)
        if self.roi_rect:
            self.roi_selection_canvas.delete(self.roi_rect)

    def draw_roi(self, event):
        if self.roi_start:
            if self.roi_rect:
                self.roi_selection_canvas.delete(self.roi_rect)
            self.roi_rect = self.roi_selection_canvas.create_rectangle(self.roi_start[0], self.roi_start[1], event.x, event.y, outline="blue")

    def end_roi(self, event):
        if self.roi_start:
            self.roi_end = (event.x, event.y)

    def confirm_roi(self):
        """确认并映射 ROI 坐标到原始尺寸。"""
        if self.roi_start and self.roi_end:
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end

            actual_width = self.original_width / self.max_scale
            actual_height = self.original_height / self.max_scale
            start_x = (self.resized_width - actual_width) / 2
            start_y = (self.resized_height - actual_height) / 2
            end_x = start_x + actual_width
            end_y = start_y + actual_height

            x1 = min(max(0, x1 - start_x), actual_width)
            x2 = min(max(0, x2 - start_x), actual_width)
            y1 = min(max(0, y1 - start_y), actual_height)
            y2 = min(max(0, y2 - start_y), actual_height)

            scale_x_actual = (end_x - start_x)/self.resized_width
            scale_y_actual = (end_y - start_y)/self.resized_height
            x1 = x1 / scale_x_actual
            x2 = x2 / scale_x_actual
            y1 = y1 / scale_y_actual
            y2 = y2 / scale_y_actual

            # 使用缩放比例将坐标映射回原始尺寸（新代码）
            roi_x1 = int(x1 * self.scale_x)
            roi_y1 = int(y1 * self.scale_y)
            roi_x2 = int(x2 * self.scale_x)
            roi_y2 = int(y2 * self.scale_y)

            self.roi = (min(roi_x1, roi_x2), min(roi_y1, roi_y2),
                        abs(roi_x2 - roi_x1), abs(roi_y2 - roi_y1))

            self.roi_window.destroy()
            self.show_roi()
        else:
            messagebox.showerror("Error", "Invalid ROI selection.")

    def set_start_frame(self):
        if self.cap:
            self.start_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.show_frame(canvas=self.start_frame_canvas)
            self.start_frame_canvas.create_text(10, 10, anchor=tk.NW, text=f"Frame: {self.start_frame}", fill="red")
            messagebox.showinfo("Start Frame Set", f"Start frame set to {self.start_frame}")

    def set_end_frame(self):
        if self.cap:
            self.end_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.show_frame(canvas=self.end_frame_canvas)
            self.end_frame_canvas.create_text(10, 10, anchor=tk.NW, text=f"Frame: {self.end_frame}", fill="red")
            messagebox.showinfo("End Frame Set", f"End frame set to {self.end_frame}")

    def save_video(self):
        if not self.video_path or self.start_frame is None or self.end_frame is None or self.start_frame >= self.end_frame:
            messagebox.showerror("Error", "Invalid start/end frame or no video loaded.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("MKV files", "*.mkv")])
        if not save_path:
            return

        # Determine the video format from the file extension
        ext = save_path.split('.')[-1].lower()
        if ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == 'mkv':
            fourcc = cv2.VideoWriter_fourcc(*'X264')
        else:
            messagebox.showerror("Error", "Unsupported file format.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        out = cv2.VideoWriter(save_path, fourcc, 10.0, (self.roi[2], self.roi[3]))

        total_frames_to_save = self.end_frame - self.start_frame + 1
        for i in range(self.start_frame, self.end_frame + 1):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                break
            cropped_frame = frame[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]
            out.write(cropped_frame)

            # Update progress
            self.progress_label.config(text=f"Saving frame {i - self.start_frame + 1}/{total_frames_to_save}")
            self.root.update_idletasks()

        out.release()
        messagebox.showinfo("Save Video", "Video saved successfully.")

    def update_frame(self):
        if self.playing:
            ret, frame = self.cap.read()
            if ret:
                self.show_frame(frame)
                self.progress.set(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            else:
                self.playing = False
                return
            self.root.after(30, self.update_frame)  # Schedule next frame update
        elif self.paused and self.roi:
            self.show_roi()

    def show_frame(self, frame=None, canvas=None):
        if frame is None and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.resize_image_to_fit_canvas(frame, self.video_canvas.winfo_width(), self.video_canvas.winfo_height())
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        if canvas:
            if canvas == self.start_frame_canvas:
                self.start_img_label = imgtk
            elif canvas == self.end_frame_canvas:
                self.end_img_label = imgtk
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        else:
            self.video_img_label = imgtk
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def show_roi(self):
        """在视频帧上显示用户选择的 ROI 区域。"""
        if self.cap and self.roi:
            ret, frame = self.cap.read()
            if ret:
                x, y, w, h = self.roi
                cropped_frame = frame[y:y + h, x:x + w]
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                # 将裁剪后的帧缩放并显示在 ROI 画布上
                cropped_frame = self.resize_image_to_fit_canvas(cropped_frame,
                                                                self.roi_canvas.winfo_width(),
                                                                self.roi_canvas.winfo_height())
                img = Image.fromarray(cropped_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.roi_img_label = imgtk
                self.roi_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def resize_image_to_fit_canvas(self, image, canvas_width, canvas_height):
        img_height, img_width = image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        # Create a white background
        background = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        # Center the resized image on the background
        y_offset = (canvas_height - new_height) // 2
        x_offset = (canvas_width - new_width) // 2
        background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
        return background

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()
