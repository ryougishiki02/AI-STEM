import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import threading
import matplotlib.pyplot as plt

def calculate_bottom_difference(frame1, frame2, num_rows=3):
    bottom_rows_frame1 = frame1[-num_rows:, :]
    bottom_rows_frame2 = frame2[-num_rows:, :]
    diff = cv2.absdiff(bottom_rows_frame1, bottom_rows_frame2)
    return np.mean(diff)

def refine_video(input_path, threshold=1, num_rows=3):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return [], None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, reference_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return [], None

    recorded_frames = [reference_frame]
    diff_values = [0]  # store difference for first frame as 0
    current_frame_index = 1

    while current_frame_index < frame_count:
        ret, current_frame = cap.read()
        if not ret:
            break
        difference = calculate_bottom_difference(reference_frame, current_frame, num_rows)
        diff_values.append(difference)
        if difference > threshold:
            recorded_frames.append(current_frame)
        reference_frame = current_frame
        current_frame_index += 1

    cap.release()
    return recorded_frames, (frame_width, frame_height), diff_values

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Refinement App")
        self.root.geometry("1200x800")

        # Store original video path for re-refine
        self.original_video_path = None

        # ====== Top control frame ======
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Threshold input
        tk.Label(control_frame, text="Threshold:").grid(row=0, column=0, sticky='e')
        self.threshold_entry = tk.Entry(control_frame, width=6)
        self.threshold_entry.insert(0, "5")
        self.threshold_entry.grid(row=0, column=1, sticky='w')

        # Num rows input
        tk.Label(control_frame, text="Num Rows:").grid(row=0, column=2, sticky='e')
        self.num_rows_entry = tk.Entry(control_frame, width=6)
        self.num_rows_entry.insert(0, "3")
        self.num_rows_entry.grid(row=0, column=3, sticky='w')

        # FPS input
        tk.Label(control_frame, text="Output FPS:").grid(row=0, column=4, sticky='e')
        self.fps_entry = tk.Entry(control_frame, width=6)
        self.fps_entry.insert(0, "10")
        self.fps_entry.grid(row=0, column=5, sticky='w')

        # Load button
        self.load_btn = ttk.Button(control_frame, text="Load Video", command=self.load_video)
        self.load_btn.grid(row=0, column=6, padx=5)

        # Apply params button
        self.apply_btn = ttk.Button(control_frame, text="Apply", command=self.apply_params, state=tk.DISABLED)
        self.apply_btn.grid(row=0, column=7, padx=5)

        # Export button
        self.export_btn = ttk.Button(control_frame, text="Export Refined Video", command=self.export_video, state=tk.DISABLED)
        self.export_btn.grid(row=0, column=8, padx=5)

        # ====== Play control frame (2nd row buttons) ======
        play_control_frame = tk.Frame(root)
        play_control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Left video controls
        self.play_left_btn = ttk.Button(play_control_frame, text="Play Left", command=self.play_left, state=tk.DISABLED)
        self.play_left_btn.grid(row=0, column=0, padx=5)
        self.pause_left_btn = ttk.Button(play_control_frame, text="Pause Left", command=self.pause_left, state=tk.DISABLED)
        self.pause_left_btn.grid(row=0, column=1, padx=5)

        # Right video controls
        self.play_right_btn = ttk.Button(play_control_frame, text="Play Right", command=self.play_right, state=tk.DISABLED)
        self.play_right_btn.grid(row=0, column=2, padx=5)
        self.pause_right_btn = ttk.Button(play_control_frame, text="Pause Right", command=self.pause_right, state=tk.DISABLED)
        self.pause_right_btn.grid(row=0, column=3, padx=5)

        # ====== Video display frames ======
        video_frame = tk.Frame(root)
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left video canvas
        self.left_canvas = tk.Canvas(video_frame, bg='black')
        self.left_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Right video canvas
        self.right_canvas = tk.Canvas(video_frame, bg='black')
        self.right_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # ====== Bottom plot frame ======
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(10, 2))
        self.ax.set_title("Frame Difference Over Time")
        self.ax.set_xlabel("Frame Index")
        self.ax.set_ylabel("Difference")
        self.line_diff, = self.ax.plot([], [], label='Difference')
        self.line_thresh = self.ax.axhline(y=0, color='r', linestyle='--', label='Threshold')
        self.ax.legend()

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.X)

        # Internal states
        self.left_frames = []
        self.right_frames = []
        self.diff_values = []
        self.left_index = 0
        self.right_index = 0
        self.left_playing = False
        self.right_playing = False
        self.output_fps = 10

        # Lock for threading
        self.lock = threading.Lock()

        # Bind resize event to update video display scale
        self.left_canvas.bind("<Configure>", self._resize_left)
        self.right_canvas.bind("<Configure>", self._resize_right)

        # Store current resized images (to prevent garbage collection)
        self.left_img_tk = None
        self.right_img_tk = None

    def load_video(self):
        path = filedialog.askopenfilename(title="Select Video File",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if not path:
            return
        self.original_video_path = path

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open selected video.")
            return

        self.left_frames.clear()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.left_frames.append(frame)
        cap.release()

        self.diff_values.clear()
        # Initial refine with current params
        self.apply_btn.config(state=tk.NORMAL)
        self.play_left_btn.config(state=tk.NORMAL)
        self.pause_left_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.DISABLED)

        self.right_frames.clear()
        self.right_index = 0
        self.left_index = 0
        self.left_playing = False
        self.right_playing = False

        self.update_plot([], 0)

        self.show_frame(self.left_canvas, self.left_frames[0])
        self.right_canvas.delete("all")

    def apply_params(self):
        if self.original_video_path is None:
            messagebox.showwarning("Warning", "Please load a video first.")
            return
        try:
            threshold = float(self.threshold_entry.get())
            num_rows = int(self.num_rows_entry.get())
            fps_input = self.fps_entry.get()
            self.output_fps = float(fps_input) if fps_input else 10
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for threshold, num rows, and fps.")
            return

        self.left_playing = False
        self.right_playing = False

        # Run refine in thread to avoid blocking UI
        threading.Thread(target=self._refine_and_update, args=(threshold, num_rows), daemon=True).start()

    def _refine_and_update(self, threshold, num_rows):
        self.export_btn.config(state=tk.DISABLED)
        self.play_right_btn.config(state=tk.DISABLED)
        self.pause_right_btn.config(state=tk.DISABLED)
        self.right_frames.clear()
        self.right_index = 0

        refined_frames, size, diff_vals = refine_video(self.original_video_path, threshold, num_rows)
        if not refined_frames:
            messagebox.showerror("Error", "Refining video failed.")
            return

        with self.lock:
            self.right_frames = refined_frames
            self.diff_values = diff_vals

        self.export_btn.config(state=tk.NORMAL)
        self.play_right_btn.config(state=tk.NORMAL)
        self.pause_right_btn.config(state=tk.NORMAL)

        # Reset right video index
        self.right_index = 0
        self.update_plot(self.diff_values, threshold)
        self.show_frame(self.right_canvas, self.right_frames[0])

    def update_plot(self, diff_values, threshold):
        self.ax.clear()
        self.ax.set_title("Frame Difference Over Time")
        self.ax.set_xlabel("Frame Index")
        self.ax.set_ylabel("Difference")
        self.ax.plot(diff_values, label="Difference")
        self.ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        self.ax.legend()
        self.canvas_plot.draw()

    def show_frame(self, canvas, frame):
        if frame is None:
            return
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            return  # 避免画布太小

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)

        img_width, img_height = pil_img.size
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2

        img_tk = ImageTk.PhotoImage(pil_img)

        # 这里不要 delete("all"), 而是判断是否已经有图像，更新图片
        if canvas == self.left_canvas:
            if not hasattr(self, 'left_img_id') or self.left_img_id is None:
                self.left_img_id = canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
            else:
                canvas.coords(self.left_img_id, x, y)
                canvas.itemconfig(self.left_img_id, image=img_tk)
            self.left_img_tk = img_tk  # 防止垃圾回收

        else:
            if not hasattr(self, 'right_img_id') or self.right_img_id is None:
                self.right_img_id = canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
            else:
                canvas.coords(self.right_img_id, x, y)
                canvas.itemconfig(self.right_img_id, image=img_tk)
            self.right_img_tk = img_tk  # 防止垃圾回收

    def play_left(self):
        if not self.left_frames:
            return
        if self.left_playing:
            return
        self.left_playing = True
        self.play_left_btn.config(state=tk.DISABLED)
        self.pause_left_btn.config(state=tk.NORMAL)
        threading.Thread(target=self._play_left_thread, daemon=True).start()

    def pause_left(self):
        self.left_playing = False
        self.play_left_btn.config(state=tk.NORMAL)
        self.pause_left_btn.config(state=tk.DISABLED)

    def _play_left_thread(self):
        while self.left_playing and self.left_index < len(self.left_frames):
            frame = self.left_frames[self.left_index]
            self.show_frame(self.left_canvas, frame)
            self.left_index += 1
            self.root.after(int(1000 / self.output_fps))
            # small sleep to yield thread and allow UI update
            import time
            time.sleep(0.01)
        self.left_playing = False
        self.play_left_btn.config(state=tk.NORMAL)
        self.pause_left_btn.config(state=tk.DISABLED)
        self.left_index = 0

    def play_right(self):
        if not self.right_frames:
            return
        if self.right_playing:
            return
        self.right_playing = True
        self.play_right_btn.config(state=tk.DISABLED)
        self.pause_right_btn.config(state=tk.NORMAL)
        threading.Thread(target=self._play_right_thread, daemon=True).start()

    def pause_right(self):
        self.right_playing = False
        self.play_right_btn.config(state=tk.NORMAL)
        self.pause_right_btn.config(state=tk.DISABLED)

    def _play_right_thread(self):
        while self.right_playing and self.right_index < len(self.right_frames):
            frame = self.right_frames[self.right_index]
            self.show_frame(self.right_canvas, frame)
            self.right_index += 1
            self.root.after(int(1000 / self.output_fps))
            import time
            time.sleep(0.01)
        self.right_playing = False
        self.play_right_btn.config(state=tk.NORMAL)
        self.pause_right_btn.config(state=tk.DISABLED)
        self.right_index = 0

    def export_video(self):
        if not self.right_frames:
            messagebox.showwarning("Warning", "No refined video to export.")
            return

        filetypes = [
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("MKV files", "*.mkv"),
            ("WMV files", "*.wmv"),
        ]

        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=filetypes)
        if not save_path:
            return

        height, width = self.right_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需求换成 'XVID' 等
        out = cv2.VideoWriter(save_path, fourcc, self.output_fps, (width, height), isColor=False)

        for frame in self.right_frames:
            # 转灰度
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            out.write(gray_frame)

        out.release()
        messagebox.showinfo("Info", f"Exported grayscale video to:\n{save_path}")

    def _resize_left(self, event):
        # On resize, redraw current left frame to fit new size
        if self.left_frames and 0 <= self.left_index < len(self.left_frames):
            self.show_frame(self.left_canvas, self.left_frames[self.left_index])

    def _resize_right(self, event):
        # On resize, redraw current right frame to fit new size
        if self.right_frames and 0 <= self.right_index < len(self.right_frames):
            self.show_frame(self.right_canvas, self.right_frames[self.right_index])

if __name__ == '__main__':
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
