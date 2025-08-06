from image.pkg import Backgroundremover, load_img

image = load_img()

# remove background
# Define the scaling parameters and remove the background from the image
remover = Backgroundremover(image, max_scale=2048, min_scale=32)
image = remover.remove()
remover.visualize()  # 可视化去背景后的图像
remover.save()       # 保存去背景后的图像