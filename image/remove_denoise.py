from image.pkg import Backgroundremover, load_img, Prediction, Average_img
from model.unet.unet import Unet as Unet5
from model.unet.unet3 import Unet as Unet3

# load Image
image = load_img()

# remove background:
# Define the scaling parameters and remove the background from the image
remover = Backgroundremover(image, max_scale=2048, min_scale=16)
image = remover.remove()
remover.visualize()  # 可视化去背景后的图像

periodicity = 3
stride = 6/8
batch_size = 8
# periodicity: number of boxes in one row or column; stride: stride of the sliding window;
# model: Unet(n_classes=1); batch_size: batch size for prediction
denoiser = Prediction(image, periodicity, stride, Unet5(n_classes=1), batch_size)
ori_image, denoised_image = denoiser.denoise()
denoiser.visualize(denoised_image)  # 可视化去噪后的图像
# denoiser.save(denoised_image)

Average_img(denoised_image, ori_image, windows_size = 768)