import numpy as np
import torch

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from gradcam import GradCam
from guided_backprop import GuidedBackprop

from guided_gradcam import guided_grad_cam
import h5py
from backend.grad_cam import GradCam as GC

# from backend.grad_cam import test

# Get params
target_example = 0  # Snake
(original_image, prep_img, target_class, file_name_to_export, pretrained_model) = \
    get_example_params(target_example)
resnet_model_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\feature_activations_fixed\resnet_model_11-18_08-32_lr_0_0001_pretrained_reg_30epochs_rod_0_1_da_10.pth'
res = torch.load(resnet_model_path)
h5_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data_longitudinal_fixed.h5'
h5_data = h5py.File(h5_path, 'r')

for k in h5_data.keys():
    if ~h5_data[k].attrs['train_data']:
        test_img = h5_data[k]
        test_label = h5_data[k].attrs['label_suvr']
        break

tl = 'ResNet.layer4.2.relu'
ol = 'ResNet.avgpool'

prep_img = prep_img.cuda()
# Grad cam
# gcv2 = GradCam(pretrained_model, target_layer=11)
# gcv2 = GC(res, conv_act_layer=tl, output_layer=ol)
pretrained_model = pretrained_model.cuda()
gcv2 = GC(pretrained_model, conv_act_layer='features.11')

# Generate cam mask
cam = gcv2.generate_cam(prep_img, target_class)
print('Grad cam completed')

# Guided backprop
GBP = GuidedBackprop(pretrained_model)
# Get gradients
guided_grads = GBP.generate_gradients(prep_img, target_class)
print('Guided backpropagation completed')

# Guided Grad cam
cam_gb = guided_grad_cam(cam, guided_grads)
save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
grayscale_cam_gb = convert_to_grayscale(cam_gb)
save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
print('Guided grad cam completed')
print('test')