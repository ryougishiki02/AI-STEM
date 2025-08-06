from unet import Unet
from pkg import data_devide, data_read, data_load, Args, train_den, save_lists_to_excel
import os
import torch

os.environ['CUDA_VISIBLE_DIVICES']='0'

net = Unet(n_classes=1, drop=0.3)
#print(net)
images_path = '../../dataset/image/'
labels_path = '../../dataset/noNoiseNoBackgroundUpinterpolation2x'
devide = False

#dataset prepare
if devide:
    data_devide(images_path, labels_path,train_ratio=0.7,select_k=[1],select_block=[9],select=True)
    print("sucessfully devide the image")

images,labels,images_test,labels_test = data_read(visual = False, vis_num = 5)

args = Args(batch_size = 8, lr = 0.001, epochs = 100,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            save_path = 'output/unet_0.001.pth', beat_path = 'output/unet_best_0.001.pth',
            resume = False, resume_path = 'output/unet_best_0.001.pth')

train_Loader,test_Loader = data_load(args)

train_psnr,test_psnr,train_loss,test_loss,train_ssim,test_ssim = train_den(net, train_Loader, test_Loader, args)

save_lists_to_excel('output/train_out.xlsx', train_1=train_psnr, test_1=test_psnr,
                    train_loss=train_loss, test_loss=test_loss, train_2 = train_ssim, test_2 = test_ssim, segment=False)

