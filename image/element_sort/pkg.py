import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

class AtomClassifierUI:
    def __init__(self, root, atoms, ori_image_np, value_key='i_mean'):
        self.root = root
        self.atoms = atoms.copy()
        self.ori_image_np = ori_image_np
        self.value_key = value_key
        self.thresholds = []

        # Tkinter UI setup
        self.root.title("Atom Classifier")

        # Main layout
        self.frame_top = ttk.Frame(root)
        self.frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.frame_bottom = ttk.Frame(root)
        self.frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # Figure for image + histogram
        self.fig, (self.ax_img, self.ax_hist) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_top)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Slider
        self.slider = tk.Scale(self.frame_bottom, from_=self.atoms[self.value_key].min(),
                               to=self.atoms[self.value_key].max(), orient=tk.HORIZONTAL,
                               resolution=0.1, label="Threshold", command=self.update_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Buttons
        self.btn_confirm = ttk.Button(self.frame_bottom, text="Confirm Threshold", command=self.confirm_threshold)
        self.btn_confirm.pack(side=tk.LEFT)

        self.btn_delete = ttk.Button(self.frame_bottom, text="Delete Threshold", command=self.delete_threshold)
        self.btn_delete.pack(side=tk.LEFT)

        self.draw_all()

    def classify_atoms(self):
        """Classify atoms based on multiple thresholds."""
        values = self.atoms[self.value_key].values
        labels = np.zeros(len(values), dtype=int)
        for i, th in enumerate(sorted(self.thresholds)):
            labels[values > th] += 1
        return labels

    def draw_all(self):
        self.ax_img.clear()
        self.ax_hist.clear()

        # --- Draw Image + Atoms ---
        self.ax_img.imshow(self.ori_image_np, cmap='gray')
        labels = self.classify_atoms()
        cmap = plt.get_cmap("tab10")

        for i, (_, row) in enumerate(self.atoms.iterrows()):
            color = cmap(labels[i] % 10)
            self.ax_img.add_patch(Circle((row['x'], row['y']), radius=row['r'], color=color, fill=True, lw=1))

        self.ax_img.set_title("Atoms Visualization")
        self.ax_img.axis('off')

        # --- Draw Histogram ---
        values = self.atoms[self.value_key]
        self.ax_hist.hist(values, bins=50, color='lightgray', edgecolor='black')
        for th in self.thresholds:
            self.ax_hist.axvline(th, color='blue', linestyle='--', linewidth=1)
        # Slider position
        self.ax_hist.axvline(self.slider.get(), color='red', linestyle='-', linewidth=2)
        self.ax_hist.set_title(f"Histogram of {self.value_key}")
        self.ax_hist.set_xlabel(self.value_key)

        self.canvas.draw()

    def update_slider(self, val):
        self.draw_all()

    def confirm_threshold(self):
        th = self.slider.get()
        if th not in self.thresholds:
            self.thresholds.append(th)
            self.draw_all()

    def delete_threshold(self):
        if self.thresholds:
            self.thresholds.pop()
            self.draw_all()
