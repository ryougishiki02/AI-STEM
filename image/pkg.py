import os
from time import time
import torch
import torchvision
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from model.unet.unet import Unet
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    gaussian_filter1d
)
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

def load_img():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    filetypes = [("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
    img_path = filedialog.askopenfilename(
        title="Open Image File",
        filetypes=filetypes
    )

    root.destroy()

    if not img_path:
        print("Cancel loading image.")
        return None

    image = Image.open(img_path).convert('L')  # 转为灰度
    print(f"Image has been read: {img_path}")
    return image

def np_pil(image_np):
    # 先转换成float64，保证数值计算准确
    arr = image_np.astype(np.float64)
    min_val = arr.min()
    max_val = arr.max()

    # 防止 max == min 导致除0
    if max_val == min_val:
        scaled = np.zeros_like(arr)
    else:
        # 线性归一化到0~255
        scaled = (arr - min_val) / (max_val - min_val) * 255
    # 转成uint8
    scaled_uint8 = scaled.astype(np.uint8)

    # 转为PIL图像，默认灰度'L'
    pil_img = Image.fromarray(scaled_uint8, mode='L')

    return pil_img

# 对图片增加灰条，防止缩放失真
def pad_img(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('L', size, color=0)
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    # 左上的坐标
    return new_image, nw, nh

class Backgroundremover:
    def __init__(self, image, max_scale = 2048, min_scale=16):
        self.image = image
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.w, self.h = image.size

    def small(self, img, scale_0, scale_1):
        while scale_0 > scale_1:
            scale_0 = scale_0/2
            img = cv2.resize(img, dsize=(int(scale_0),int(scale_0))) #dsize宽度在前
        return img

    def large(self, img,scale_0,scale_1):
        while scale_0 < scale_1:
            scale_0 = scale_0*2
            img = cv2.resize(img, dsize=(int(scale_0),int(scale_0)))
        return img

    def remove(self):
        image, nw, nh = pad_img(self.image, (self.max_scale, self.max_scale))
        image = np.array(image).astype(np.float32) / 255.0  # Convert to numpy array and normalize to [0, 1]

        small_img = self.small(image, self.max_scale, self.min_scale)
        large_img = self.large(small_img, self.min_scale, self.max_scale)
        output = np_pil(image - large_img)

        box = (int((self.max_scale - nw) // 2),
               int((self.max_scale - nh) // 2),
               int(self.max_scale - (self.max_scale - nw) // 2),
               int(self.max_scale - (self.max_scale - nh) // 2))
        # print(box, output.size)
        self.output = output.crop(box)

        return self.output

    def visualize(self, figsize=(5, 5), dpi=300):
        plt.figure(figsize = figsize, dpi = dpi)
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()

    def save(self):
        root = tk.Tk()
        root.withdraw()  # 不显示主窗口
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=filetypes,
            title="Save Image As..."
        )
        root.destroy()
        if not path:
            print("Cancel saving image.")
            return
        # 自动提取扩展名并判断格式
        ext = os.path.splitext(path)[-1].lower().strip('.')
        format_map = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
            'tif': 'TIFF',
            'tiff': 'TIFF',
            'bmp': 'BMP'
        }
        format = format_map.get(ext)
        self.output.save(path, format=format)
        print(f"Image have been save to {path}")

class Average_img():
    def __init__(self, image1, image2, windows_size=768):
        self.img1_np = np.array(image1.convert("L")).astype(np.float32)
        self.img2_np = np.array(image2.convert("L")).astype(np.float32)
        self.windows_size = windows_size

        self.root = tk.Tk()
        self.root.title("Averaged image")

        # 输入框和标签：图片1叠加次数
        tk.Label(self.root, text="Image 1 times:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.times1_var = tk.StringVar(value="1")
        self.entry1 = ttk.Entry(self.root, textvariable=self.times1_var, width=10)
        self.entry1.grid(row=0, column=1, padx=5, pady=5)

        # 输入框和标签：图片2叠加次数
        tk.Label(self.root, text="Image 2 times:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.times2_var = tk.StringVar(value="1")
        self.entry2 = ttk.Entry(self.root, textvariable=self.times2_var, width=10)
        self.entry2.grid(row=1, column=1, padx=5, pady=5)

        # 显示按钮，点击后更新显示
        self.btn = ttk.Button(self.root, text="Show Averaged Image", command=self.show_average)
        self.btn.grid(row=2, column=0, padx=5, pady=10, sticky='ew')
        # 保存按钮
        self.save_btn = ttk.Button(self.root, text="Save Image", command=self.save_image, state="disabled")
        self.save_btn.grid(row=2, column=1, padx=5, pady=10, sticky='ew')

        # 图像显示区域
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=3, column=0, columnspan=2, pady=10)

        self.root.mainloop()

    def show_average(self):
        try:
            times1 = int(self.times1_var.get())
            times2 = int(self.times2_var.get())
            if times1 < 1 or times2 < 1:
                raise ValueError
        except ValueError:
            tk.messagebox.showerror("Input Error", "Please enter valid positive integers for times.")
            return

        avg_np = (self.img1_np * times1 + self.img2_np * times2) / (times1 + times2)
        avg_img = Image.fromarray(avg_np.astype(np.uint8))
        self.avg_img = avg_img

        # 自动缩放：最大宽/高不超过 400
        w, h = avg_img.size
        scale = min(self.windows_size / w, self.windows_size / h, 1.0)  # <=1.0 缩小图像，不放大
        new_size = (int(w * scale), int(h * scale))
        avg_img_resized = avg_img.resize(new_size, Image.BILINEAR)

        imgtk = ImageTk.PhotoImage(avg_img_resized)
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk
        self.save_btn.config(state="normal")  # 使保存按钮可用

    def save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif"), ("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Averaged Image"
        )
        if not path:
            return
        try:
            # 自动识别格式
            ext = path.split('.')[-1].lower()
            format_map = {
                'jpg': 'JPEG', 'jpeg': 'JPEG',
                'png': 'PNG',
                'tif': 'TIFF', 'tiff': 'TIFF'
            }
            fmt = format_map.get(ext, 'PNG')  # 默认 PNG
            self.avg_img.save(path, format=fmt)
            messagebox.showinfo("Saved", f"Image saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

class Prediction:
    def __init__(self, image, periodicity, stride = 7/8, model = Unet(),
                 batch_size = 8):
        self.ori_image = image
        self.image, self.nw, self.nh = pad_img(image, (256 * periodicity, 256 * periodicity))
        self.periodicity = periodicity
        self.stride = stride
        self.model = model
        self.batch_size = batch_size
        self.box_size = 256
        self.transform1 = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                         torchvision.transforms.ToTensor()])
        self.transform2 = torchvision.transforms.ToPILImage()

    def model_load(self, title="Select Model File"):
        # load model
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("PyTorch model files", "*.pth *.pt"), ("All files", "*.*")]
        )
        root.destroy()

        if not model_path:
            print("No model file selected.")
            return None

        start_time = time()

        self.model_path = model_path
        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.eval()
        print("Model loaded from:", model_path)

        return start_time

    def segment(self,):
        start_time = self.model_load(title="Select A Model For Segmentation")

        stride = int(self.box_size * self.stride)
        if self.periodicity == 1:
            d_stride = 0
        else:
            d_stride = int((self.box_size - stride)/2)
        full_size = self.box_size * self.periodicity
        output = Image.new('L', (full_size, full_size), color=0)
        patches = []
        positions = []

        # Step 1: Extract patches and store positions
        tops = list(range(0, full_size - self.box_size + 1, stride))
        lefts = list(range(0, full_size - self.box_size + 1, stride))
        # 补一个贴边 patch，如果最后一个 patch 没有覆盖边缘
        if tops[-1] + self.box_size < full_size:
            tops.append(full_size - self.box_size)
        if lefts[-1] + self.box_size < full_size:
            lefts.append(full_size - self.box_size)
        for top in tops:
            for left in lefts:
                box = (left, top, left + self.box_size, top + self.box_size)
                patch = self.image.crop(box).resize((256, 256), Image.BICUBIC)
                patch = self.transform1(patch)
                patches.append(patch)
                positions.append((left, top))

        print('Number of patches:', len(patches))
        # Step 2: Batch inference
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch_patches = patches[i:i + self.batch_size]
                input = torch.stack(batch_patches).cuda()  # shape: (B, 1, 256, 256)

                preds = self.model(input)  # shape: (B, C, 256, 256)
                preds = torch.argmax(preds, dim=1).float().cpu()  # shape: (B, 256, 256)

                for j in range(preds.shape[0]):
                    pred_img = self.transform2(preds[j])  # Tensor → PIL
                    pred_img = pred_img.resize((self.box_size, self.box_size), Image.NEAREST)

                    d_left, d_right, d_top, d_bottom = 0, 0, 0, 0
                    left, top = positions[i + j]

                    if left == 0:
                        d_left = -d_stride
                    elif left >= full_size - self.box_size:
                        d_right = d_stride
                    if top == 0:
                        d_top = -d_stride
                    elif top >= full_size - self.box_size:
                        d_bottom = d_stride

                    # 裁剪中心区域
                    cropped = pred_img.crop((
                        d_stride + d_left, d_stride + d_top,
                        self.box_size - d_stride + d_right, self.box_size - d_stride + d_bottom
                    ))

                    paste_box = (
                        left + d_stride + d_left, top + d_stride + d_top,
                        left + self.box_size - d_stride + d_right, top + self.box_size - d_stride + d_bottom
                    )
                    output.paste(cropped, paste_box)

        # 将output的大小调整为原尺寸
        box = (int((self.box_size * self.periodicity - self.nw) // 2),
               int((self.box_size * self.periodicity - self.nh) // 2),
               int(self.box_size * self.periodicity - (self.box_size * self.periodicity - self.nw) // 2),
               int(self.box_size * self.periodicity - (self.box_size * self.periodicity - self.nh) // 2))
        # print(box,output.size)
        output = output.crop(box).resize((self.ori_image.size[0], self.ori_image.size[1]), Image.NEAREST)
        end_time = time()
        print('Total segmentation cost time: {} s; Number of patches:{}'.format(end_time - start_time, len(patches)))

        return self.ori_image, output

    def denoise(self):
        start_time = self.model_load(title="Select A Model For Denoising")

        stride = int(self.box_size * self.stride)
        if self.periodicity == 1:
            d_stride = 0
        else:
            d_stride = int((self.box_size - stride)/2)
        full_size = self.box_size * self.periodicity
        output = Image.new('L', (full_size, full_size), color=0)

        patches = []
        positions = []
        # Step 1: Extract patches and store positions
        tops = list(range(0, full_size - self.box_size + 1, stride))
        lefts = list(range(0, full_size - self.box_size + 1, stride))
        # 补一个贴边 patch，如果最后一个 patch 没有覆盖边缘
        if tops[-1] + self.box_size < full_size:
            tops.append(full_size - self.box_size)
        if lefts[-1] + self.box_size < full_size:
            lefts.append(full_size - self.box_size)
        for top in tops:
            for left in lefts:
                box = (left, top, left + self.box_size, top + self.box_size)
                patch = self.image.crop(box).resize((256, 256), Image.BICUBIC)
                patch = self.transform1(patch)
                patches.append(patch)
                positions.append((left, top))

        # Step 2: Batch inference
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch_patches = patches[i:i + self.batch_size]
                input = torch.stack(batch_patches).cuda()  # shape: (B, 1, 256, 256)

                preds = self.model(input)  # shape: (B, C, 256, 256)
                preds = preds.squeeze(1).detach().cpu().float()  # shape: (B, 256, 256)

                for j in range(preds.shape[0]):
                    pred_img = self.transform2(preds[j])  # Tensor → PIL
                    pred_img = pred_img.resize((self.box_size, self.box_size), Image.BICUBIC)

                    d_left, d_right, d_top, d_bottom = 0, 0, 0, 0
                    left, top = positions[i + j]

                    if left == 0:
                        d_left = -d_stride
                    elif left >= full_size - self.box_size:
                        d_right = d_stride
                    if top == 0:
                        d_top = -d_stride
                    elif top >= full_size - self.box_size:
                        d_bottom = d_stride

                    # 裁剪中心区域
                    cropped = pred_img.crop((
                        d_stride + d_left, d_stride + d_top,
                        self.box_size - d_stride + d_right, self.box_size - d_stride + d_bottom
                    ))

                    paste_box = (
                        left + d_stride + d_left, top + d_stride + d_top,
                        left + self.box_size - d_stride + d_right, top + self.box_size - d_stride + d_bottom
                    )
                    output.paste(cropped, paste_box)

        # 将output的大小调整为原尺寸
        box = (int((self.box_size * self.periodicity - self.nw) // 2),
               int((self.box_size * self.periodicity - self.nh) // 2),
               int(self.box_size * self.periodicity - (self.box_size * self.periodicity - self.nw) // 2),
               int(self.box_size * self.periodicity - (self.box_size * self.periodicity - self.nh) // 2))
        # print(box,output.size)
        output = output.crop(box).resize((self.ori_image.size[0], self.ori_image.size[1]), Image.BICUBIC)
        end_time = time()
        print('Total denoise cost time: {} s; Number of patches:{}'.format(end_time - start_time, len(patches)))

        return self.ori_image, output

    def visualize(self, image, figsize=(5, 5), dpi=300):
        plt.figure(figsize = figsize, dpi = dpi)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def save(self, image):
        root = tk.Tk()
        root.withdraw()  # 不显示主窗口
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=filetypes,
            title="Save Image As..."
        )
        root.destroy()
        if not path:
            print("Cancel saving image.")
            return
        # 自动提取扩展名并判断格式
        ext = os.path.splitext(path)[-1].lower().strip('.')
        format_map = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
            'tif': 'TIFF',
            'tiff': 'TIFF',
            'bmp': 'BMP'
        }
        format = format_map.get(ext)
        image.save(path, format=format)
        print(f"Image have been save to {path}")
        image.save(path, format = format)

class Position:
    def __init__(self, ori_img, binary, erosion_iter= 1, split_method = 'floodsill',
                    min_distance=5, filter_ratio = 0.5):
        self.ori_img = ori_img
        self.binary = binary
        self.erosion_iter = erosion_iter
        self.split_method = split_method
        self.min_distance = min_distance
        self.filter_ratio = filter_ratio

    def fit_circle(self, xy):
        #代数圆拟合法（Kåsa方法）
        x = xy[:, 0]
        y = xy[:, 1]
        A = np.c_[2*x, 2*y, np.ones_like(x)]
        b = x**2 + y**2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        r = np.sqrt(c + xc**2 + yc**2)
        return xc, yc, r

    def split_atoms(self):
        if self.split_method == 'floodfill':
            # 这里的连通性可以根据实际情况调整，1表示4连通，2表示8连通
            labeled = label(self.binary, connectivity=1)
            regions = regionprops(labeled)
        elif self.split_method == 'watershed':
            """使用距离变换 + 分水岭切分粘连区域"""
            distance = distance_transform_edt(self.binary)
            local_max = peak_local_max(
                distance,
                labels=self.binary,
                min_distance=self.min_distance,
                footprint=np.ones((3, 3)),
                exclude_border=False
            )
            local_max_mask = np.zeros_like(self.binary, dtype=bool)
            local_max_mask[tuple(local_max.T)] = True
            markers = label(local_max_mask)
            labels_ws = watershed(-distance, markers, mask=self.binary)

            regions = regionprops(labels_ws)
        else:
            raise ValueError("Invalid split_method. Choose 'floodfill' or 'watershed'.")

        return regions

    def position_detect(self):
        if self.erosion_iter:
            # Morphology: Erosion
            binary = binary_erosion(self.binary, iterations=self.erosion_iter)
            # binary = binary_dilation(binary, iterations=erosion_iter)

        regions = self.split_atoms()

        ori_areas = []
        atoms = []  # 保存所有区域数据
        H, W = self.ori_img.shape
        # 提取所有区域信息
        for region in regions:
            ori_area = region.area
            coords = region.coords[:, ::-1]  # skimage 是 (row, col), 转为 (x, y)
            xc, yc, r = self.fit_circle(coords)
            fit_area = np.pi * r * r

            # --- 优化：只在圆的最小包围盒内计算掩膜 ---
            x_min = max(int(np.floor(xc - r)), 0)
            x_max = min(int(np.ceil(xc + r)), W - 1)
            y_min = max(int(np.floor(yc - r)), 0)
            y_max = min(int(np.ceil(yc + r)), H - 1)
            region_img = self.ori_img[y_min:y_max + 1, x_min:x_max + 1]
            Y, X = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
            circle_mask = (X - xc) ** 2 + (Y - yc) ** 2 <= r ** 2
            region_values = region_img[circle_mask]
            intensity_mean = region_values.mean() if region_values.size > 0 else 0
            intensity_max = region_values.max() if region_values.size > 0 else 0

            # 保存结果
            atoms.append({
                'x': xc,
                'y': yc,
                'r': r,
                'fit_area': fit_area,
                'ori_area': ori_area,
                'i_mean': intensity_mean,
                'i_peak': intensity_max
            })
            ori_areas.append(ori_area)
        atoms_df = pd.DataFrame(atoms)

        # 计算原始面积的中位数
        median_area = np.median(ori_areas)
        # 保留面积在 [1-filter * ave_area, 1+filter * ave_area] 范围内的
        if self.filter_ratio:
            atoms_df = atoms_df[
                (atoms_df['fit_area'] >= median_area * (1 - self.filter_ratio)) &
                (atoms_df['fit_area'] <= median_area * (1 + self.filter_ratio))
                ].copy()

        return atoms_df

    def visualize(self, atoms_df, save_path = None, figsize=(15, 5), dpi=300):
        # 绘制原图和拟合的圆
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        # 子图 1：binary 图
        axs[0].imshow(self.binary, cmap='gray')
        axs[0].axis('off')
        # 子图 2：拟合圆
        axs[1].imshow(self.ori_img, cmap='gray')
        plt.axis('off')
        axs[2].imshow(self.binary, cmap='gray')
        plt.axis('off')
        print("检测到{}个原子".format(len(atoms_df)))
        for _, atom in atoms_df.iterrows():
            xc, yc, r = atom['x'], atom['y'], atom['r']
            axs[1].add_patch(plt.Circle((xc, yc), r, color='r', fill=False))
            axs[2].add_patch(plt.Circle((xc, yc), r, color='r', fill=False))
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print("Results saved to:", save_path)
        plt.show()

    def save(self, atoms_df):
        # 创建并隐藏 Tkinter 主窗口
        root = tk.Tk()
        root.withdraw()
        # 文件保存对话框
        filetypes = [
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",  # 默认扩展名
            filetypes=filetypes,
            title="Save Atoms Data As..."
        )
        root.destroy()
        if not path:
            print("Cancel saving CSV.")
            return
        # 判断扩展名（可选，防止误操作）
        ext = os.path.splitext(path)[-1].lower()
        if ext not in [".csv", ".txt"]:
            print(f"Unsupported file extension: {ext}")
            return
        try:
            atoms_df.to_csv(path, index=False)
            print(f"CSV file saved to: {path}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")

def draw_atoms(img, atoms, element = True, colors = ['red', 'blue', 'orange', 'green'], figsize = (8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    if element:
        """在原图上用不同颜色标注不同 group 的原子"""
        for _, row in atoms.iterrows():
            group = row['group']
            color = colors[int(group)]
            circle = plt.Circle((row['x'], row['y']), row['r'], edgecolor=color,
                                fill=False, linewidth=1.2)
            ax.add_patch(circle)
    else:
        for _, row in atoms.iterrows():
            circle = plt.Circle((row['x'], row['y']), row['r'], color='red',
                                fill=False, linewidth=1.2)
            ax.add_patch(circle)
    plt.show()

class Element:
    def __init__(self, ori_img, atoms, I = 'i_peak', start_bins = 200, bin_step = 20,
                 min_prominence_ratio = 0.25, smooth_sigma = 5):
        self.ori_img = ori_img
        self.atoms = atoms
        self.I = I
        self.start_bins = start_bins
        self.bin_step = bin_step
        self.min_prominence_ratio = min_prominence_ratio
        self.smooth_sigma = smooth_sigma

    def element_detect(self):
        i_vals = self.atoms[self.I].values
        bin_count = self.start_bins
        iter_peaks = [[], []]
        # 动态调整 bins，找最多峰
        while True:
            counts, bins = np.histogram(i_vals, bins=bin_count)
            smoothed = gaussian_filter1d(counts.astype(float), sigma=self.smooth_sigma)
            peaks, _ = find_peaks(smoothed)
            max_height = smoothed[peaks].max()
            valid_peaks = peaks[smoothed[peaks] >= max_height * self.min_prominence_ratio]

            # 如果连续两次没有新峰，就停止
            if len(valid_peaks) == len(iter_peaks[-2]):
                self.final_peaks = valid_peaks
                self.final_bins = bins
                self.final_counts = counts
                self.final_smoothed = smoothed
                break
            iter_peaks.append(valid_peaks)
            bin_count += self.bin_step

        # 获取峰之间的 valley（作为边界）
        valley_indices = []
        for i in range(len(self.final_peaks) - 1):
            left = self.final_peaks[i]
            right = self.final_peaks[i + 1]
            valley = np.argmin(self.final_smoothed[left:right]) + left
            valley_indices.append(valley)
        # 用 bin 边界值表示分割点
        self.split_values = [self.final_bins[i + 1] for i in valley_indices]

        # 给每个原子分组
        def assign_group(val):
            for i, thresh in enumerate(self.split_values):
                if val < thresh:
                    return i
            return len(self.split_values)

        self.atoms['group'] = self.atoms[self.I].apply(assign_group)

        return self.atoms

    def visualize(self, save_path = None, colors = ['red', 'blue', 'orange', 'green'], dpi=300):
        bin_centers = (self.final_bins[:-1] + self.final_bins[1:]) / 2
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, self.final_counts, label='Histogram', alpha=0.5)
        plt.plot(bin_centers, self.final_smoothed, label='Smoothed', linewidth=2)
        plt.scatter(bin_centers[self.final_peaks], self.final_smoothed[self.final_peaks], color='red', label='Peaks')
        for split in self.split_values:
            plt.axvline(split, color='gray', linestyle='--', alpha=0.6)
        plt.xlabel("I")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

        draw_atoms(self.ori_img, self.atoms, element=True,
                   colors=colors, figsize=(8, 8))
