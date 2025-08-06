from image.pkg import Backgroundremover, Prediction, Position, Element, load_img
from model.unet.unet import Unet as Unet
from model.unet.unet3 import Unet as Unet3
import numpy as np

image = load_img()

#remove background
remover = Backgroundremover(image, max_scale=2048, min_scale=32)
image = remover.remove()
remover.visualize()  # 可视化去背景后的图像

periodicity = 1
stride = 6/8
batch_size = 8
# periodicity: number of boxes in one row or column; stride: stride of the sliding window;
# batch_size: batch size for prediction
denoiser = Prediction(image, periodicity, stride, Unet(n_classes=1), batch_size)
ori_image, image = denoiser.denoise()
denoiser.visualize(image)  # 可视化去噪后的图像

segmenter = Prediction(image, periodicity, stride, Unet(), batch_size)
image, output = segmenter.segment()
segmenter.visualize(output)  # 可视化分割后的图像

#二值化,检测原子位置
output_np = np.array(output.convert("L")).astype(np.uint8)
ori_image_np = np.array(ori_image.convert("L")).astype(np.float32)
#atoms为一个DataFrame：x, y, r, fit_area, ori_area, i_mean, i_peak
#split_method = 'watershed'/'floodfill'
position_detecter = Position(ori_image_np, output_np, erosion_iter = 0,
                             split_method = 'floodfill', min_distance = 5,filter_ratio = 0.8)
atoms = position_detecter.position_detect()
position_detecter.visualize(atoms, save_path = None)

# 检测元素
element_detecter = Element(ori_image_np, atoms, I = 'i_peak', start_bins=20, bin_step=10,
                           min_prominence_ratio=0.25, smooth_sigma = 3)
atoms = element_detecter.element_detect()
element_detecter.visualize(save_path = None,
                           colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown', 'pink'])
