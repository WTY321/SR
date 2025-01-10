
import torch
from torchvision import transforms
from PIL import Image
from utils.util_cam import get_cam, resize_cam, blend_cam
import torchvision.utils as vutils
_GREEN = (18, 217, 15)
_RED = (15, 18, 217)
IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

def phototest(cam,img,box,gt_bbox,iou,name):
    image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))
    cam = cam.cpu().numpy().transpose(0, 2, 3, 1)
    image_ = img.clone().detach().cpu() * image_mean + image_std
    image_ = image_.numpy().transpose(0, 2, 3, 1)
    image_ = image_[:, :, :, ::-1] * 255
    cam_ = resize_cam(cam[0], size=(256,256))
    blend, heatmap = blend_cam(image_[0], cam_)
    (x0, y0, x1, y1) = box
    cv2.rectangle(blend, (int(x0), int(y0)), (int(x1), int(y1)), _GREEN, thickness=4)
    (x2, y2, x3, y3) = gt_bbox
    cv2.rectangle(blend, (int(x2), int(y2)), (int(x3), int(y3)), _RED, thickness=4)
    cv2.putText(blend, '%.1f' % (iou * 100), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    blend=blend[:, :, ::-1] / 255.
    blend=blend.transpose(2,0,1)
    blend_tensor=torch.tensor(blend)
    savename=name.split('/')[0]+'-'+name.split('/')[1].split('.')[0]+'.jpg'
    vutils.save_image(blend_tensor,'result/'+savename)

if __name__ == '__main__':
    phototest('E:/ACoL_pytorch/CUB_200_2011/CUB_200_2011/images/200.Common_Yellowthroat/Common_Yellowthroat_0045_190563.jpg')






