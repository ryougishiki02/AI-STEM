from model.unet.unet import Unet as Unet
from model.unet.unet3 import Unet as Unet3
from image.pkg import Backgroundremover, Prediction, Position, load_img
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import tkinter as tk
from tkinter import simpledialog,ttk, filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from collections import defaultdict,Counter
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

def crop_image(image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原图
    ax1.imshow(image, cmap='gray')
    ax1.set_title("original image")
    ax1.axis('off')  # 不显示坐标轴

    cropped_img = None  # 用来存储裁剪后的图像
    crop_coords = [None, None, None, None]  # [x1, y1, x2, y2]

    # 当选择区域时，裁剪图像并更新 cropped_img
    def onselect(eclick, erelease):
        nonlocal cropped_img
        # 获取裁剪区域的左上角 (x1, y1) 和右下角 (x2, y2)
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # 使用 PIL 的 crop 方法来裁剪图像
        cropped_img = image.crop((x1, y1, x2, y2))
        crop_coords[:] = [x1, y1, x2, y2]  # 更新裁剪坐标

        # 在第二个子图上显示裁剪后的图像
        ax2.clear()  # 清空ax2的内容
        ax2.imshow(cropped_img, cmap='gray')
        ax2.set_title('cropped image')
        ax2.axis('off')  # 不显示坐标轴
        plt.draw()  # 更新显示

    # 创建矩形选择器
    rect_selector = RectangleSelector(ax1, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels')

    # 按钮点击事件，点击后退出交互并关闭窗口
    def on_button_click(event):
        plt.close(fig)  # 关闭窗口，结束交互

    # 添加按钮
    ax_button = plt.axes([0.8, 0.05, 0.15, 0.075])  # 设置按钮位置 [x, y, width, height]
    button = Button(ax_button, 'comfirm')  # 按钮标题
    button.on_clicked(on_button_click)  # 为按钮绑定点击事件

    plt.show()

    return cropped_img, crop_coords  # 返回裁剪后的图像和坐标

class PixelFilter:
    result = None  # class variable to hold result

    def __init__(self, root, image):
        self.root = root
        self.root.title("Pixel Filter")

        # Load image
        self.image = image
        self.image_np = np.array(self.image)
        self.filtered_image_np = self.image_np.copy()

        self.min_val = 0
        self.max_val = 255

        self.tk_img = None  # to keep reference
        self.canvas_width = 512
        self.canvas_height = 512

        self.setup_widgets()

    def setup_widgets(self):
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Histogram
        fig, self.ax = plt.subplots(figsize=(4, 3))
        self.hist_canvas = FigureCanvasTkAgg(fig, master=self.left_frame)
        self.hist_canvas.get_tk_widget().pack()
        self.update_histogram()

        self.slider_min = tk.Scale(self.left_frame, from_=0, to=255,
                                   orient=tk.HORIZONTAL, label="Min Value",
                                   length=200, command=self.on_slider_change)
        self.slider_min.set(self.min_val)
        self.slider_min.pack(fill="x", pady=5)

        self.slider_max = tk.Scale(self.left_frame, from_=0, to=255,
                                   orient=tk.HORIZONTAL, label="Max Value",
                                   length=200, command=self.on_slider_change)
        self.slider_max.set(self.max_val)
        self.slider_max.pack(fill="x", pady=5)

        confirm_btn = tk.Button(self.left_frame, text="Confirm", command=self.confirm_selection)
        confirm_btn.pack(pady=10)

        self.canvas = tk.Canvas(self.right_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.update_display()

    def update_histogram(self):
        self.ax.clear()
        self.ax.hist(self.image_np.ravel(), bins=256, range=(0, 255), color='gray')
        self.ax.set_title("Pixel Value Distribution")
        self.ax.set_xlabel("Pixel Value")
        self.ax.set_ylabel("Frequency")

        # 这里传入单个数值，不要传列表
        self.min_line = self.ax.axvline(x=self.min_val, color='red', linestyle='--', linewidth=1)
        self.max_line = self.ax.axvline(x=self.max_val, color='red', linestyle='--', linewidth=1)

        self.hist_canvas.draw()

    def on_slider_change(self, event=None):
        self.min_val = self.slider_min.get()
        self.max_val = self.slider_max.get()

        if self.min_val > self.max_val:
            self.max_val = self.min_val
            self.slider_max.set(self.max_val)

        if self.min_line and self.max_line:
            self.min_line.set_xdata([self.min_val, self.min_val])
            self.max_line.set_xdata([self.max_val, self.max_val])
            self.hist_canvas.draw()

        mask = (self.image_np >= self.min_val) & (self.image_np <= self.max_val)
        self.filtered_image_np = np.where(mask, self.image_np, 0)
        self.update_display()

    def update_display(self):
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            return

        img_pil = Image.fromarray(self.filtered_image_np.astype(np.uint8)).copy()
        img_w, img_h = img_pil.size
        scale = min(self.canvas_width / img_w, self.canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_width // 2,
                                 self.canvas_height // 2,
                                 anchor=tk.CENTER,
                                 image=self.tk_img)

    def on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.update_display()

    def confirm_selection(self):
        PixelFilter.result = Image.fromarray(self.filtered_image_np.astype(np.uint8))
        self.root.destroy()

    @classmethod
    def run_with_image(cls, image):
        root = tk.Tk()
        app = cls(root, image)
        root.mainloop()
        return cls.result

class AtomFilter:
    def __init__(self, root, atoms_df, ori_image, I='i_peak'):
        self.root = root
        self.root.title("Intensity Filter")

        self.atoms_df = atoms_df
        self.ori_image = ori_image  # PIL.Image.Image
        self.filtered_atoms = atoms_df  # 初始为全部

        # 计算 i_peak 的最小值和最大值
        self.I = I
        self.i_peak_min = float(atoms_df[self.I].min())
        self.i_peak_max = float(atoms_df[self.I].max())

        # 初始化滑动条变量
        self.min_var = tk.DoubleVar(value=self.i_peak_min)
        self.max_var = tk.DoubleVar(value=self.i_peak_max)

        # 添加监听器（变量变更后自动调用 _apply_filter）
        self.min_var.trace_add("write", lambda *args: self._apply_filter())
        self.max_var.trace_add("write", lambda *args: self._apply_filter())

        # 创建布局容器
        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.control_frame = tk.Frame(root)
        self.control_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # 设置响应式布局权重
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        self._draw_histogram()
        self._draw_controls()
        self._show_overlay_image()  # 初始显示

    def _draw_histogram(self):
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(4, 3), dpi=100)

        self.hist_ax.hist(self.atoms_df[self.I], bins=50, color='steelblue')
        self.hist_ax.set_title("Intensity Histogram")
        self.hist_ax.set_xlabel("Intensity (I)")
        self.hist_ax.set_ylabel("Count")
        self.hist_fig.tight_layout()

        # 添加红色 min/max 线
        self.min_line = self.hist_ax.axvline(x=self.min_var.get(), color='red', linestyle='--', label='Min I')
        self.max_line = self.hist_ax.axvline(x=self.max_var.get(), color='red', linestyle='--', label='Max I')

        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.left_frame)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_controls(self):
        # 允许 column=1 自动拉伸
        self.control_frame.columnconfigure(1, weight=1)

        # Min slider
        tk.Label(self.control_frame, text="Min I:").grid(row=0, column=0, sticky="w")
        min_slider = tk.Scale(self.control_frame, variable=self.min_var,
                              from_=self.i_peak_min, to=self.i_peak_max,
                              orient=tk.HORIZONTAL, resolution=0.01)
        min_slider.grid(row=0, column=1, sticky="ew")

        # Max slider
        tk.Label(self.control_frame, text="Max I:").grid(row=1, column=0, sticky="w")
        max_slider = tk.Scale(self.control_frame, variable=self.max_var,
                              from_=self.i_peak_min, to=self.i_peak_max,
                              orient=tk.HORIZONTAL, resolution=0.01)
        max_slider.grid(row=1, column=1, sticky="ew")

        # Confirm/Close
        close_btn = ttk.Button(self.control_frame, text="Close & Return", command=self._close_and_return)
        close_btn.grid(row=0, column=2, rowspan=2, padx=10)

    def _apply_filter(self):
        min_val = self.min_var.get()
        max_val = self.max_var.get()
        if min_val > max_val:
            return  # 忽略非法范围

        # 更新红线位置，传入长度为2的列表
        self.min_line.set_xdata([min_val, min_val])
        self.max_line.set_xdata([max_val, max_val])
        self.hist_canvas.draw()

        # 过滤数据并刷新右图
        mask = (self.atoms_df[self.I] >= min_val) & (self.atoms_df[self.I] <= max_val)
        self.filtered_atoms = self.atoms_df[mask]
        self._show_overlay_image()

    def _show_overlay_image(self):
        # 清除右边图像
        for widget in self.right_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.imshow(self.ori_image, cmap='gray')

        xs = self.filtered_atoms['x']
        ys = self.filtered_atoms['y']
        ax.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')

        ax.set_title("Selected Atoms Overlay")
        ax.axis('off')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _close_and_return(self):
        self.root.quit()
        self.root.destroy()

    def get_filtered_atoms(self):
        return self.filtered_atoms, self.min_var.get(), self.max_var.get()

def Denosie_Position(image, periodicity=4, stride=6/8, batch_size=8, model = 'unet',
                     split_method='watershed', min_distance=5, filter_ratio=0.8,  I='i_peak',save = False):
    print()
    print("********** Start denoising and position detection **********")
    if model == 'unet':
        denoiser = Prediction(image, periodicity, stride, Unet(n_classes=1), batch_size)
    elif model == 'unet3':
        denoiser = Prediction(image, periodicity, stride, Unet3(n_classes=1), batch_size)
    ori_image, denoised_image = denoiser.denoise()
    denoiser.visualize(denoised_image)  # 可视化去噪后的图像

    # 像素滤波
    filtered_image = PixelFilter.run_with_image(denoised_image)
    # Crop an area as Standard area
    filtered_cropped_image, croped_coords = crop_image(filtered_image)
    # Crop the original denoised image using the same coordinates
    cropped_image = denoised_image.crop(croped_coords)
    periodicity_cropped = math.ceil((cropped_image.size[0] / denoised_image.size[0]) * periodicity)

    # Segment and Predict position
    if model == 'unet':
        segmenter1 = Prediction(filtered_image, periodicity, stride, Unet(), batch_size)
        segmenter2 = Prediction(filtered_cropped_image, periodicity_cropped, stride, Unet(), batch_size)
    elif model == 'unet3':
        segmenter1 = Prediction(filtered_image, periodicity, stride, Unet3(), batch_size)
        segmenter2 = Prediction(filtered_cropped_image, periodicity_cropped, stride, Unet3(), batch_size)
    image1, output1 = segmenter1.segment()  # 分割原图像
    image2, output2 = segmenter2.segment()  # 分割裁剪后的图像
    segmenter1.visualize(output1)  # 可视化分割后的图像
    segmenter2.visualize(output2)  # 可视化裁剪后的图像
    # 二值化,检测原子位置
    output_np1 = np.array(output1.convert("L")).astype(np.uint8)
    image_np1 = np.array(denoised_image.convert("L")).astype(np.float32)
    output_np2 = np.array(output2.convert("L")).astype(np.uint8)
    image_np2 = np.array(cropped_image.convert("L")).astype(np.float32)
    # atoms为一个DataFrame：x, y, r, fit_area, ori_area, i_mean, i_peak
    # split_method = 'watershed'/'floodfill'
    position_detecter1 = Position(image_np1, output_np1, erosion_iter=0,
                                  split_method=split_method, min_distance=min_distance,
                                  filter_ratio=min_distance)
    position_detecter2 = Position(image_np2, output_np2, erosion_iter=0,
                                  split_method=split_method, min_distance=min_distance,
                                  filter_ratio=filter_ratio)
    atoms1 = position_detecter1.position_detect()
    atoms2 = position_detecter2.position_detect()
    # 可视化原子位置
    position_detecter1.visualize(atoms1, save_path=None)
    position_detecter2.visualize(atoms2, save_path=None)

    root = tk.Tk()
    atomfilter = AtomFilter(root, atoms1, denoised_image, I)
    root.mainloop()
    flitered_atoms1, min_I, max_I = atomfilter.get_filtered_atoms()
    flitered_atoms2 = atoms2[(atoms2[I] >= min_I) & (atoms2[I] <= max_I)]

    print('Filtered atoms in original image: {}, Filtered atoms in cropped image: {}'
          .format(len(flitered_atoms1),len(flitered_atoms2)))

    if save:
        denoiser.save(denoised_image)
        segmenter1.save(output1)
        segmenter2.save(output2)
        position_detecter1.save(atoms1)
        position_detecter2.save(atoms2)

    print("********** Denoising and position detection completed **********")
    print()

    return (denoised_image, filtered_image, cropped_image, filtered_cropped_image,
            flitered_atoms1[['x', 'y']], flitered_atoms2[['x', 'y']])

class Histogramfilter:
    def __init__(self, root, all_sorted_distances, image=None, positions=None):
        self.root = root
        self.root.title("Distance Histogram & Threshold Selector")

        self.distances = np.array(all_sorted_distances)
        self.image = image
        self.positions = positions  # shape: (N, 2)
        self.selected_threshold = tk.DoubleVar()
        self.bin_count = tk.IntVar(value=50)
        self.threshold_line = None
        self.circles = []

        # GUI layout
        self.frame_plot = tk.Frame(root)
        self.frame_plot.pack(fill=tk.BOTH, expand=True)

        self.frame_controls = tk.Frame(root)
        self.frame_controls.pack(fill=tk.X, pady=10)

        self._draw_controls()
        self._draw_histogram()

        if self.image is not None and self.positions is not None:
            self._draw_circle_overlay()

    def _draw_controls(self):
        # bin数量控制
        tk.Label(self.frame_controls, text="Number of bins:").pack(side=tk.LEFT)
        bin_entry = tk.Spinbox(self.frame_controls, from_=5, to=500, increment=1,
                               textvariable=self.bin_count, width=5,
                               command=self._update_histogram)
        bin_entry.pack(side=tk.LEFT, padx=5)

        # 滑条控制 threshold
        tk.Label(self.frame_controls, text="Threshold:").pack(side=tk.LEFT, padx=(20, 0))

        # 新建滑条容器，使其占据剩余空间
        slider_frame = tk.Frame(self.frame_controls)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.slider = tk.Scale(
            slider_frame,
            from_=self.distances.min(),
            to=self.distances.max(),
            orient=tk.HORIZONTAL,
            resolution=0.01,
            variable=self.selected_threshold,
            command=lambda e: self._update_threshold_display()
        )
        self.slider.pack(fill=tk.X, expand=True)

        # 当前值显示
        self.value_label = tk.Label(self.frame_controls, text="0.00")
        self.value_label.pack(side=tk.LEFT, padx=5)

        self.selected_threshold.trace_add("write", self._update_value_label)

        # 确认按钮
        ttk.Button(self.frame_controls, text="Confirm", command=self._confirm).pack(side=tk.RIGHT, padx=10)

    def _draw_histogram(self):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.hist_ax = ax
        self.hist_fig = fig

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._update_histogram()

    def _update_histogram(self):
        bins = self.bin_count.get()
        self.hist_ax.clear()
        self.hist_ax.hist(self.distances, bins=bins, color='skyblue', edgecolor='black')
        self.hist_ax.set_title("All Sorted Distances Histogram")
        self.hist_ax.set_xlabel("Distance")
        self.hist_ax.set_ylabel("Count")
        self._draw_threshold_line()
        self.hist_fig.tight_layout()
        self.canvas.draw()

    def _draw_threshold_line(self):
        threshold = self.selected_threshold.get()
        if self.threshold_line:
            self.threshold_line.remove()
        self.threshold_line = self.hist_ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)

    def _draw_circle_overlay(self):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.overlay_ax = ax
        self.overlay_fig = fig

        ax.imshow(self.image, cmap='gray')
        ax.set_title("Threshold Circles on Image")
        ax.axis('off')

        self.overlay_canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        self.overlay_canvas.draw()
        self.overlay_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._update_overlay_circles()

    def _update_overlay_circles(self):
        threshold = self.selected_threshold.get()

        for c in self.circles:
            c.remove()
        self.circles.clear()

        if self.positions is not None:
            for (x, y) in self.positions:
                solid_circle = plt.Circle((x, y), 3, color='red', fill=True)
                circle = plt.Circle((x, y), radius=threshold, edgecolor='red',
                                    facecolor='none', linewidth=2)
                self.overlay_ax.add_patch(circle)
                self.overlay_ax.add_patch(solid_circle)
                self.circles.append(circle)
                self.circles.append(solid_circle)

        self.overlay_canvas.draw()

    def _update_threshold_display(self):
        self._draw_threshold_line()
        self.canvas.draw()
        if self.image is not None and self.positions is not None:
            self._update_overlay_circles()

    def _update_value_label(self, *args):
        val = self.selected_threshold.get()
        self.value_label.config(text=f"{val:.2f}")

    def _confirm(self):
        self.root.quit()
        self.root.destroy()

    def get_selected_threshold(self):
        return self.selected_threshold.get()

def Calculate_neighbor(positions, r, image, if_crop=False):
    tree = cKDTree(positions)
    all_distances = []
    all_positions = []

    # 遍历每个原子，查询其半径 r 内的邻居
    for i, pos in enumerate(positions):
        # 查询 r 范围内的邻居索引（包括自己）
        indices = tree.query_ball_point(pos, r)

        # 排除自身
        indices = [j for j in indices if j != i]
        if not indices:
            all_distances.append(np.array([]))
            all_positions.append(np.array([]))
            continue

        neighbor_positions = positions[indices]
        distances = np.linalg.norm(neighbor_positions - pos, axis=1)

        # 排序
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_positions = neighbor_positions[sorted_indices]

        all_distances.append(sorted_distances)
        all_positions.append(sorted_positions)

    # 提取所有原子的排序距离，组合成一个序列
    all_sorted_distances = []
    for sorted_distances in all_distances:
        all_sorted_distances.extend(sorted_distances)

    if if_crop:
        return all_distances, all_positions, None

    root = tk.Tk()
    random_indices = np.random.choice(len(positions), size=3, replace=False)
    random_positions = positions[random_indices]
    histogramselector = Histogramfilter(root, all_sorted_distances, image, random_positions)
    root.mainloop()
    cut_off = histogramselector.get_selected_threshold()

    print('********** Atoms Classify **********')
    print("Selected cut-off distance:", cut_off)

    return all_distances, all_positions, cut_off

class DBSCANClusterGUI:
    def __init__(self, vectors_list):
        self.data = np.array(vectors_list)
        self.labels = None
        self.cluster_centers = None
        self.canvas_widget = None
        self.fig = None

        self.window = tk.Tk()
        self.window.title("DBSCAN for Vector")

        self._build_widgets()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.mainloop()

    def _build_widgets(self):
        # Eps slider
        ttk.Label(self.window, text="Eps (density):").grid(row=0, column=0)
        self.eps_var = tk.DoubleVar(value=0.5)
        eps_slider = ttk.Scale(self.window, from_=0.1, to=2.0, variable=self.eps_var,
                               orient="horizontal", command=self._update_eps_label)
        eps_slider.grid(row=0, column=1)
        self.eps_label = ttk.Label(self.window, text=f"{self.eps_var.get():.2f}")
        self.eps_label.grid(row=0, column=2)

        # Min samples slider
        ttk.Label(self.window, text="Min Samples:").grid(row=1, column=0)
        self.min_samples_var = tk.IntVar(value=5)
        min_samples_slider = ttk.Scale(self.window, from_=1, to=10,
                                       variable=self.min_samples_var, orient="horizontal",
                                       command=self._update_min_label)
        min_samples_slider.grid(row=1, column=1)
        self.min_label = ttk.Label(self.window, text=f"{self.min_samples_var.get()}")
        self.min_label.grid(row=1, column=2)

        # Buttons
        refine_btn = ttk.Button(self.window, text="Refine", command=self._plot_clusters)
        refine_btn.grid(row=3, column=0, columnspan=2)

        save_btn = ttk.Button(self.window, text="Save Image", command=self._save_plot)
        save_btn.grid(row=3, column=2)

    def _update_eps_label(self, val):
        self.eps_label.config(text=f"{float(val):.2f}")

    def _update_min_label(self, val):
        self.min_label.config(text=f"{int(float(val))}")

    def _plot_clusters(self):
        eps = self.eps_var.get()
        min_samples = self.min_samples_var.get()

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = dbscan.fit_predict(self.data)

        # 聚类中心计算
        self.cluster_centers = []
        for label in set(self.labels):
            if label == -1:
                continue
            cluster_points = self.data[self.labels == label]
            center = np.mean(cluster_points, axis=0)
            self.cluster_centers.append(center)

        # 清除旧图像
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()

        # 创建子图
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 所有数据
        ax1.scatter(self.data[:, 0], self.data[:, 1], c='gray', s=10, alpha=0.75,
                    edgecolor='black', label='All data')
        for center in self.cluster_centers:
            ax1.scatter(center[0], center[1], s=100, c='red', marker='X')
            r = np.linalg.norm(center)
            theta = np.arctan2(center[1], center[0]) * 180 / np.pi
            ax1.text(center[0] + 0.5, center[1] + 0.5, f"r={r:.2f}\nθ={theta:.2f}°",
                     color='red', fontsize=10, fontweight='bold')
        ax1.set_title("All data")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend()

        # 分类后的数据（排除噪声点）
        if np.any(self.labels != -1):
            ax2.scatter(self.data[self.labels != -1, 0],
                        self.data[self.labels != -1, 1],
                        c=self.labels[self.labels != -1], cmap='viridis',
                        s=10, edgecolor='black', alpha=0.75, label='Classified data')
        ax2.set_title("Classified data")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend()

        # 嵌入 Tkinter
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().grid(row=2, column=0, columnspan=3)

    def _save_plot(self):
        if self.fig:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if path:
                self.fig.savefig(path)
                print(f"图像已保存到：{path}")
            else:
                print("用户取消保存")

    def _on_close(self):
        self.window.quit()
        self.window.destroy()

    def get_result(self):
        return self.cluster_centers, self.labels

def classify_atoms(label_list):
    """
    对原子根据其标签列表进行分类

    参数:
    - label_list: 每个原子的标签列表 [[labels_atom1], [labels_atom2], ...]

    返回:
    - 每个原子的分类标签列表 ['A', 'B', 'A', ...]
    """
    # 将每个原子的标签列表排序，并转为 tuple 作为特征 key
    key_list = [tuple(sorted(atom)) for atom in label_list]

    # 建立 key -> indices 的映射
    key_to_indices = defaultdict(list)
    for i, key in enumerate(key_list):
        key_to_indices[key].append(i)

    # 按出现频率排序（频率高的在前）
    sorted_keys = sorted(key_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

    # 初始化分类字典
    most_common_key = sorted_keys[0][0]
    classified = {most_common_key: 'A'}
    seen_labels = set(most_common_key)

    # 用于存储每个符号对应的元素集合
    symbol_map = {}
    label_counter = ord('B')

    # 处理其余的key
    for key, _ in sorted_keys[1:]:
        key_set = set(key)

        # 判断是否为新类别
        if key_set <= seen_labels:
            # 完全是已见过的标签，归为 A
            classified[key] = 'A'
        else:
            # 检查是否与已有 symbol 产生交集
            matched = False
            for symbol, elements in symbol_map.items():
                if key_set & elements:  # 有交集
                    classified[key] = symbol
                    matched = True
                    break

            if not matched:
                # 创建新的分类标签
                label = chr(label_counter)
                classified[key] = label
                symbol_map[label] = key_set
                label_counter += 1

            # 更新已见过的标签集合
            seen_labels.update(key_set)

    # 返回每个原子的分类标签
    return [classified[key] for key in key_list]

def Vector_centers(atoms, neighbor_distances, neighbor_positions, cutoff):
    vectors_list = []
    for i, atom in enumerate(atoms):
        distances_i = neighbor_distances[i]
        positions_i = neighbor_positions[i]
        neighbors_idx = np.where(distances_i <= cutoff)[0]
        vectors_i = positions_i[neighbors_idx] - atom
        vectors_list.append(vectors_i)

    vectors = np.concatenate(vectors_list)
    gui = DBSCANClusterGUI(vectors)
    cluster_centers, labels = gui.get_result()

    # 拆分labels为 label_list
    label_list = []
    idx = 0
    for sublist in vectors_list:
        n = len(sublist)
        label_list.append(labels[idx:idx+n])
        idx += n

    # 根据label_list对原子分类
    atoms_labels = classify_atoms(label_list)

    print('The number of vectors:', len(set(atoms_labels)))

    return cluster_centers, atoms_labels

def calculate_min_distance(centers):
    """
    计算聚类中心两两之间的最小距离

    参数:
    - centers: 聚类中心数组 [n_clusters, 2]

    返回:
    - 最小距离值
    """
    if len(centers) < 2:
        return 0

    centers = np.array(centers)
    n_centers = len(centers)

    # 使用广播计算所有中心点对之间的距离
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # 创建掩码排除对角线元素（自己与自己的距离）
    mask = ~np.eye(n_centers, dtype=bool)

    return np.min(distances[mask])

def assign_vectors_to_clusters(vectors, cluster_centers, threshold):
    """
    将向量分配给最近的聚类中心，处理标签冲突
    参数:
    - vectors: 向量数组 [n_vectors, 2]
    - cluster_centers: 聚类中心 [n_clusters, 2]
    - threshold: 距离阈值

    返回:
    - labels: 每个向量的聚类标签（-1表示不归类）
    """
    n_vectors = len(vectors)
    if n_vectors == 0:
        return []

    vectors = np.array(vectors)
    cluster_centers = np.array(cluster_centers)

    # 批量计算所有向量到所有聚类中心的距离
    # distances[i, j] 表示第j个向量到第i个聚类中心的距离
    distances = np.linalg.norm(
        cluster_centers[:, np.newaxis, :] - vectors[np.newaxis, :, :],
        axis=2
    )

    # 找到每个向量最近的聚类中心
    closest_cluster_indices = np.argmin(distances, axis=0)
    closest_distances = np.min(distances, axis=0)

    # 初始化标签和距离记录
    labels = [-1] * n_vectors
    assigned_distances = [np.inf] * n_vectors

    # 处理每个向量的分配
    for j in range(n_vectors):
        closest_idx = closest_cluster_indices[j]
        closest_dist = closest_distances[j]

        # 只有距离小于阈值才考虑分配
        if closest_dist <= threshold:
            # 检查该聚类标签是否已被其他向量使用
            existing_vector_idx = None
            for k in range(j):
                if labels[k] == closest_idx:
                    existing_vector_idx = k
                    break

            if existing_vector_idx is not None:
                # 如果当前向量距离更近，则替换
                if closest_dist < assigned_distances[existing_vector_idx]:
                    labels[existing_vector_idx] = -1
                    assigned_distances[existing_vector_idx] = np.inf
                    labels[j] = closest_idx
                    assigned_distances[j] = closest_dist
                # 否则当前向量不分配（保持-1）
            else:
                # 该聚类标签未被使用，直接分配
                labels[j] = closest_idx
                assigned_distances[j] = closest_dist

    return labels

def classify_vectors(cluster_centers, positions, all_distances, all_positions, r, dist_multi=1.5):
    """
    对每个原子的近邻向量分类，输出 vectors_list 和 labels_list

    参数:
    - cluster_centers: DBSCAN得到的聚类中心数组，形状 [n_clusters, 2]
    - positions: 所有原子的位置 [N, 2]
    - all_distances: 所有原子到其邻居的距离 [N, M]
    - all_positions: 所有原子的邻居原子位置 [N, M, 2]
    - r: 搜索半径
    - dist_multi: 距离倍数阈值参数，用于确定向量归属聚类的最大允许距离
                  threshold = min_cluster_distance * dist_multi
                  较小的值(如1.0)会更严格，只有非常接近聚类中心的向量才会被分类
                  较大的值(如2.0)会更宽松，允许距离聚类中心较远的向量也被分类
    返回:
    - vectors_list: 每个原子的向量列表
    - atoms_labels: 每个原子的分类标签
    """
    if not cluster_centers or len(positions) == 0:
        return [], []

    cluster_centers = np.array(cluster_centers)
    positions = np.array(positions)

    # 计算用于阈值判断的最小聚类间距
    min_cluster_dist = calculate_min_distance(cluster_centers)
    threshold = min_cluster_dist * dist_multi

    vectors_list = []
    labels_list = []

    # 处理每个原子
    for i, atom in enumerate(positions):
        # 找到半径 r 内的邻居
        neighbor_mask = all_distances[i] <= r
        neighbor_positions = all_positions[i][neighbor_mask]

        # 计算近邻向量
        vectors = neighbor_positions - atom

        # 将向量分配给聚类中心
        labels = assign_vectors_to_clusters(vectors, cluster_centers, threshold)

        vectors_list.append(vectors)
        labels_list.append(labels)

    # 对原子进行分类
    atoms_labels = classify_atoms(labels_list)

    return vectors_list, labels_list, atoms_labels

class AtomSelector:
    def __init__(self, vectors_list, label_list, positions, atom_labels, image_pil, marker_size=5):
        plt.close('all')
        self.vectors_list = vectors_list
        self.label_list = label_list
        self.positions = positions
        self.atom_labels = atom_labels
        self.original_image = image_pil.copy()
        self.marker_size = marker_size
        self.selected_labels = set()
        self.result = None

        self.label_counts = Counter(atom_labels)
        self.sorted_labels = [item[0] for item in self.label_counts.most_common()]
        self.label_colors = self._assign_colors(self.sorted_labels)

        self.root = tk.Tk()
        self.root.title("Atom Classification Selector")
        self._build_interface()
        self.root.mainloop()

    def _assign_colors(self, labels):
        cmap = ['green', 'pink', 'blue', 'yellow', 'red', 'orange', 'purple', 'cyan']
        return {lbl: cmap[i % len(cmap)] for i, lbl in enumerate(labels)}

    def _build_interface(self):
        self.root.geometry("1000x800")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # Canvas for image
        self.canvas = tk.Canvas(self.main_frame, bg='white')
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_resize)

        # Add UI below
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=1, column=0, pady=5, sticky="ew")

        # Label buttons
        self.label_buttons = {}
        for i, lbl in enumerate(self.sorted_labels):
            chk = ttk.Checkbutton(control_frame, text=f"{lbl} ({self.label_counts[lbl]})",
                                  command=self._update_selection)
            chk.grid(row=0, column=i, padx=5)
            self.label_buttons[lbl] = chk

        # Legend
        legend_frame = ttk.Frame(self.root)
        legend_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(legend_frame, text="Label Colors:").pack(side="left")
        for lbl in self.sorted_labels:
            color = self.label_colors[lbl]
            patch = tk.Canvas(legend_frame, width=15, height=15)
            patch.create_rectangle(0, 0, 15, 15, fill=color)
            patch.pack(side="left", padx=2)
            ttk.Label(legend_frame, text=str(lbl)).pack(side="left", padx=5)

        # Confirm/cancel
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=3, column=0, pady=5)
        ttk.Button(btn_frame, text="Confirm", command=self._confirm).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side="left", padx=10)

    def _on_resize(self, event):
        # 保持图像等比缩放，并清除所有旧图层
        canvas_w, canvas_h = event.width, event.height
        img_w, img_h = self.original_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)  # 等比缩放比例

        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_image = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)

        # 清除画布
        self.canvas.delete("all")

        # 重新添加图像
        self.canvas.create_image(0, 0, image=self.tk_image, anchor='nw')

        # 重绘原子
        self._draw_atoms((new_w, new_h))

        # 重新设置滚动区域
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

    def _draw_atoms(self, canvas_size):
        img_w, img_h = self.original_image.size
        scale_x = canvas_size[0] / img_w
        scale_y = canvas_size[1] / img_h

        for pos, lbl in zip(self.positions, self.atom_labels):
            x, y = pos[0] * scale_x, pos[1] * scale_y
            r = self.marker_size
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.label_colors[lbl], outline="")

    def _update_selection(self):
        self.selected_labels = {
            lbl for lbl, btn in self.label_buttons.items() if btn.instate(["selected"])
        }

    def _confirm(self):
        if not self.selected_labels:
            messagebox.showwarning("No Selection", "Please select at least one label.")
            return
        indices = [i for i, lbl in enumerate(self.atom_labels) if lbl in self.selected_labels]
        selected_vectors = [self.vectors_list[i] for i in indices]
        selected_label_list = [self.label_list[i] for i in indices]
        selected_positions = [self.positions[i] for i in indices]
        self.result = (selected_vectors, selected_label_list, selected_positions)
        self.root.destroy()

    def _cancel(self):
        self.result = None
        self.root.destroy()

    def get_result(self):
        return self.result

def standard_vector(image, cluster_centers):
    def initial_vector(image):
        # 显示图像
        fig, ax = plt.subplots()
        # 添加英文标题
        fig.suptitle('Please select a point and two vectors')
        ax.imshow(image)

        # 定义一个列表来存储用户选择的点
        selected_points = []
        # 用于存储绘制的点的线条对象
        drawn_points = []

        def onclick(event):
            if event.inaxes == ax:
                x, y = int(event.xdata), int(event.ydata)
                selected_points.append((x, y))
                point = ax.plot(x, y, 'ro')  # 在点击的位置绘制一个红色圆点
                drawn_points.append(point[0])  # 存储绘制的点的线条对象
                fig.canvas.draw()

        # 连接鼠标点击事件
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # 创建确认按钮
        ax_confirm = plt.axes([0.7, 0.05, 0.1, 0.075])
        btn_confirm = Button(ax_confirm, 'Confirm')

        # 创建返回上一步按钮
        ax_back = plt.axes([0.81, 0.05, 0.1, 0.075])
        btn_back = Button(ax_back, 'Back')

        def on_confirm(event):
            if len(selected_points) == 3:
                fig.canvas.mpl_disconnect(cid)
                plt.close()

        def on_back(event):
            if selected_points:
                # 移除最后一个选择的点
                selected_points.pop()
                # 移除图上对应的点
                if drawn_points:
                    last_point = drawn_points.pop()
                    last_point.remove()
                    fig.canvas.draw()

        btn_confirm.on_clicked(on_confirm)
        btn_back.on_clicked(on_back)

        plt.show()

        # 提取用户选择的点
        point_0 = np.array(selected_points[0])
        point_1 = np.array(selected_points[1])
        point_2 = np.array(selected_points[2])

        # 计算两个向量
        vector_1 = point_1 - point_0
        vector_2 = point_2 - point_0

        print(f"Selected point: {point_0}")
        print(f"Vector 1: {vector_1}")
        print(f"Vector 2: {vector_2}")

        return point_0, vector_1, vector_2

    point_0, vector_1, vector_2 = initial_vector(image)

    def find_closest_vector(target_vector, vectors):
        distances = np.linalg.norm(vectors - target_vector, axis=1)
        closest_index = np.argmin(distances)
        closest_vector = vectors[closest_index]
        return closest_vector, closest_index

    standard_a, a_index = find_closest_vector(vector_1, cluster_centers)
    standard_b, b_index = find_closest_vector(vector_2, cluster_centers)

    return standard_a, standard_b, a_index, b_index


def calculate_strain(vectors_list, label_list, standard_a, standard_b, label_a, label_b):
    print()
    print('********** Calculate Strain **********')
    strain = []

    def find_value_index(ary, value):
        if value in ary:
            return np.where(ary == value)  # 返回值的索引
        return -1  # 如果值不存在，则返回 -1

    def cal_e(a, b, u, v):
        a = a.reshape(2)
        b = b.reshape(2)
        u = u.reshape(2)
        v = v.reshape(2)
        # 构造矩阵 A 和向量 U
        A = np.array([a, b])   # 构造矩阵 [[a_x, a_y], [b_x, b_y]]
        UV_y = np.array([u[1], v[1]]).flatten()  # 构造列向量 [u_y, v_y]
        UV_x = np.array([u[0], v[0]]).flatten()  # 构造列向量 [u_y, v_y]
        # 计算 A 的逆矩阵
        # print(A,UV_x,UV_y)
        A_inv = np.linalg.inv(A)
        # 计算结果
        E_y = np.dot(A_inv, UV_y)  # 计算矩阵乘法 A^(-1) * UV
        E_x = np.dot(A_inv, UV_x)  # 计算矩阵乘法 A^(-1) * UV
        eyx, eyy = E_y
        exx, exy = E_x
        shear_strain = (exy + eyx) / 2
        lattice_rot = (exy - eyx) / 2
        emax = (exx + eyy) / 2 + np.sqrt(((exx - eyy) / 2) ** 2 + shear_strain ** 2)
        emin = (exx + eyy) / 2 - np.sqrt(((exx - eyy) / 2) ** 2 + shear_strain ** 2)
        theta = np.degrees(0.5 * np.arctan(2 * shear_strain / (exx - eyy)))
        strain = [eyx, eyy, exx, exy, emax, emin, theta]
        return strain

    for num_atom, (vector_atom_i, label_atom_i) in enumerate(zip(vectors_list, label_list)):
        index_a = find_value_index(label_atom_i, label_a)
        index_b = find_value_index(label_atom_i, label_b)
        index_no = find_value_index(label_atom_i, -1)        # 判断是否存在无法分类的向量
        if index_a != -1 and index_b != -1:
            a = vector_atom_i[index_a]
            b = vector_atom_i[index_b]
            u = a - standard_a
            v = b - standard_b
            strain_i = cal_e(a, b, u, v)
            strain.append(strain_i)
        else:
            strain_i = []
            strain.append(strain_i)

    return strain

def plot_strain(image, position, strain, a, b, view_atom=False, markersize=3):
    height, width = image.size

    def triangle_centroid(pos, a, b):
        A = pos + a
        B = pos + b
        return ((A[0] + B[0] + pos[0]) / 3, (A[1] + B[1] + pos[1]) / 3)

    valid_positions = []
    eyx, eyy, exx, exy, emax, emin, theta = [], [], [], [], [], [], []

    for i, s in enumerate(strain):
        if s:
            center = triangle_centroid(position[i], a, b)
            valid_positions.append(center)
            eyx.append(s[0])
            eyy.append(s[1])
            exx.append(s[2])
            exy.append(s[3])
            emax.append(s[4])
            emin.append(s[5])
            theta.append(s[6])

    if not valid_positions:
        raise ValueError("No valid strain data to plot.")

    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    xx, yy = np.meshgrid(x, y)

    ave_vector = (np.linalg.norm(a) + np.linalg.norm(b)) / 2

    # 初始 colorbar 范围
    e_all = np.abs(np.array(eyx + eyy + exx + exy + emax + emin))
    default_max = np.max(e_all) * 0.8

    def interpolate(data):
        return griddata(valid_positions, data, (xx, yy), method='cubic')

    strain_components = {
        'yx': eyx,
        'yy': eyy,
        'xx': exx,
        'xy': exy,
        'max': emax,
        'min': emin
    }

    def draw_strain_maps(vmax):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        cmap = plt.cm.coolwarm
        norm = Normalize(vmin=-vmax, vmax=vmax)

        for ax, (title, data) in zip(axes.flat, strain_components.items()):
            interp = interpolate(data)
            im = ax.imshow(interp, cmap=cmap, norm=norm, origin='upper')
            ax.set_title(f"e{title}")
            ax.axis('off')

        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, orientation='vertical')
        cbar.set_label("Strain Value")
        plt.show()

    def draw_principal_strain(vmax):
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(valid_positions)):
            x, y = valid_positions[i]
            angle = np.deg2rad(theta[i])
            len_max = np.abs(emax[i] / vmax) * ave_vector
            len_min = np.abs(emin[i] / vmax) * ave_vector
            x_max = len_max * np.cos(angle) / 2
            y_max = len_max * np.sin(angle) / 2
            x_min = len_min * np.cos(angle + np.pi / 2) / 2
            y_min = len_min * np.sin(angle + np.pi / 2) / 2
            ax.plot([x - x_max, x + x_max], [y - y_max, y + y_max], color='red', lw=1)
            ax.plot([x - x_min, x + x_min], [y - y_min, y + y_min], color='blue', lw=1)
        ax.set_title('Principal Strain')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.show()

    # 初次绘图
    draw_strain_maps(default_max)
    draw_principal_strain(default_max)

    # 支持多次交互更新 colorbar
    while True:
        try:
            user_input = input("\nEnter new color bar max value (or press Enter to exit): ")
            if user_input.strip().lower() in ["", "q", "quit", "exit"]:
                print("Exiting interactive color bar adjustment.")
                break
            new_max = float(user_input.strip())
            draw_strain_maps(new_max)
            draw_principal_strain(new_max)
        except ValueError:
            print("Invalid input. Please enter a numeric value.")