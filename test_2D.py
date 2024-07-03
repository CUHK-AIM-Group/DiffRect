import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from dataloaders.dataset import MRSEG19Normalization

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./datasets/ACDC', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument('--ckpt' , type=str, default='./logs/ACDC/diffrect_7_labeled/unet/unet_best_model.pth',
                    help='checkpoint_name', required=True)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        jc = metric.binary.jc(pred, gt)
        return dice, jc, hd95, asd
    else:
        return 0, 0, 50, 10.

def test_single_volume(case, net, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    # normalize if max value is greater than 1
    if image.max() > 1:
        image = MRSEG19Normalization()(image, mode='Max_Min')
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main) > 1:
                out_main = out_main[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    return first_metric, second_metric, third_metric


def Inference(FLAGS, list=None):
    with open(FLAGS.root_path + '/'+list+'.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    test_save_path = FLAGS.ckpt.split('/unet/')[0] + '/predictions/'
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    # print(torch.load(FLAGS.ckpt).keys())
    if 'state_dict' in torch.load(FLAGS.ckpt).keys():
        info = net.load_state_dict(torch.load(FLAGS.ckpt)['state_dict'])
    else:
        info = net.load_state_dict(torch.load(FLAGS.ckpt))
    print("init weight from {}".format(FLAGS.ckpt))
    print(info)
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    if 'ACDC' in FLAGS.root_path:
        list_list = ['val', 'test']
    else:
        list_list = ['val']
    for list in list_list:
        result_metric = Inference(FLAGS, list)
        print("Results per class")
        print(result_metric)
        print(" | Dice | HD95 | Jaccard | ASD |")
        # average over result_metric[0] ... result_metric[FLAGS.num_classes - 2]
        print("Average")
        res = np.zeros_like(result_metric[0])
        for i in range(FLAGS.num_classes - 1):
            res += result_metric[i]
        res /= (FLAGS.num_classes - 1)
        print(res)
