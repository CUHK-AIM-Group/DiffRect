import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (
    BaseDataSets,
    TwoStreamBatchSampler,
    WeakStrongAugment_Ours,
)
from networks.net_factory import net_factory
from networks.unet_de import UNet_LDMV2
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume_refinev2 as test_single_volume
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./datasets/ACDC", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="ACDC/diffrect", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=6, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
parser.add_argument("--img_channels", type=int, default=1, help="images channels, 1 if ACDC, 3 if GLAS")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="confidence threshold for using pseudo-labels",
)

parser.add_argument("--labeled_bs", type=int, default=3, help="labeled_batch_size per gpu")
parser.add_argument("--labeled_num", type=int, default=7, help="labeled data")
parser.add_argument("--refine_start", type=int, default=1000, help="start iter for rectification")
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
# rf
parser.add_argument("--base_chn_rf", type=int, default=64, help="rect model base channel")
parser.add_argument("--ldm_beta_sch", type=str, default='cosine', help="diffusion schedule beta")
parser.add_argument("--ts", type=int, default=10, help="ts")
parser.add_argument("--ts_sample", type=int, default=2, help="ts_sample")
parser.add_argument("--ref_consistency_weight", type=float, default=-1, help="consistency_rampup")
parser.add_argument("--no_color", default=False, action="store_true", help="no color image")
parser.add_argument("--no_blur", default=False, action="store_true", help="no blur image")
parser.add_argument("--rot", type=int, default=359, help="rotation angle")

args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "1": 32,
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif 'Task05' in dataset:
        assert args.num_classes == 3
        if 'split1' in dataset:
            ref_dict = {'2': 30}
        elif 'split2' in dataset:
            ref_dict = {'2': 40}
    elif 'mrseg19' in dataset:
        ref_dict = {'7': 103}
    else:
        raise NotImplementedError
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



def train(args, snapshot_path):
    args_dict = vars(args)
    for key, val in args_dict.items():
        logging.info("{}: {}".format(str(key), str(val)))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False, in_chns=1):
        model = net_factory(net_type=args.model, in_chns=in_chns, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def get_comp_loss(weak, strong, bs=args.batch_size):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            comp_loss, as_weight
        """
        il_output = torch.reshape(
            strong,
            (
                bs,
                args.num_classes,
                args.patch_size[0] * args.patch_size[1],
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(args.patch_size[0] * args.patch_size[1]))
        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss, as_weight

    def normalize(tensor):
        min_val = tensor.min(1, keepdim=True)[0]
        max_val = tensor.max(1, keepdim=True)[0]
        result = tensor - min_val
        result = result / max_val
        return result

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment_Ours(args.patch_size, args)]),
        # transform=transforms.Compose([WeakStrongAugment_Ours(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    db_test = BaseDataSets(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    logging.info("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    model = create_model(in_chns=args.img_channels)
    model_dict = {}
    model_dict['base_chn'] = args.base_chn_rf
    print("INPUT CHANNELS:", 3+args.img_channels)
    refine_model = UNet_LDMV2(in_chns=3+args.img_channels, class_num=num_classes, out_chns=num_classes, ldm_method='replace', ldm_beta_sch=args.ldm_beta_sch, ts=args.ts, ts_sample=args.ts_sample).cuda()

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    refine_optimizer = optim.SGD(refine_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # if restoring previous models:
    if args.load:
        try:
            # check if there is previous progress to be restored:
            logging.info(f"Snapshot path: {snapshot_path}")
            iter_num = []
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename:
                    basename, extension = os.path.splitext(filename)
                    iter_num.append(int(basename.split("_")[2]))
            iter_num = max(iter_num)
            for filename in os.listdir(snapshot_path):
                if "model_iter" in filename and str(iter_num) in filename:
                    model_checkpoint = filename
        except Exception as e:
            logging.warning(f"Error finding previous checkpoints: {e}")

        try:
            logging.info(f"Restoring model checkpoint: {model_checkpoint}")
            model, optimizer, start_epoch, performance = util.load_checkpoint(
                snapshot_path + "/" + model_checkpoint, model, optimizer
            )
            logging.info(f"Models restored from iteration {iter_num}")
        except Exception as e:
            logging.warning(f"Unable to restore model checkpoint: {e}, using new model")

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # Define color mappings for each class
    if args.num_classes == 4:
        color_map = {
            0: (0, 0, 0),    # background class
            1: (255, 0, 0),  # class 1 (red)
            2: (0, 255, 0),  # class 2 (green)
            3: (0, 0, 255),  # class 3 (blue)
        }
    elif args.num_classes == 3:
        color_map = {
            0: (0, 0, 0),    # background class
            1: (255, 0, 0),  # class 1 (red)
            2: (0, 255, 0),  # class 2 (green)
        }
    elif args.num_classes == 2:
        color_map = {
            0: (0, 0, 0),    # background class
            1: (255, 255, 255),  # class 1 (white)
        }

    # set to train
    model.train()
    refine_model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            # replace the label of unlabeled data with pure black to avoid influence
            label_batch[args.labeled_bs:] = torch.zeros_like(label_batch[args.labeled_bs:])

            # outputs for model
            outputs_weak = model(weak_batch)
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
            outputs_strong = model(strong_batch)
            outputs_strong_soft = torch.softmax(outputs_strong, dim=1)

            # minmax normalization for softmax outputs before applying mask
            pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
            outputs_weak_masked = outputs_weak_soft * pseudo_mask
            pseudo_outputs = torch.argmax(outputs_weak_masked.detach(), dim=1, keepdim=False)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            comp_loss, as_weight = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft)
            # supervised loss
            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            # unsupervised loss
            unsup_loss = (
                ce_loss(outputs_strong[args.labeled_bs :], pseudo_outputs[args.labeled_bs :])
                + dice_loss(outputs_strong_soft[args.labeled_bs :], pseudo_outputs[args.labeled_bs :].unsqueeze(1))
                + as_weight * comp_loss
            )

            ##############################################################################
            # generate strong pseudo labels
            pseudo_mask_strong = (normalize(outputs_strong_soft) > args.conf_thresh).float()
            outputs_strong_masked = outputs_strong_soft * pseudo_mask_strong
            pseudo_outputs_strong = torch.argmax(outputs_strong_masked.detach(), dim=1, keepdim=False) # lab+unlab

            # (a) Label Semantic Encoding
            # (a) 1. encode weak pseudo labels
            pseudo_outputs_for_refine = pseudo_outputs.detach().clone()  # lab+unlab
            pseudo_outputs_numpy = pseudo_outputs_for_refine.clone().detach().cpu().numpy()
            pseudo_outputs_color = pl_weak_embed(color_map, pseudo_outputs_numpy)

            # (a) 2. encode strong pseudo labels
            pseudo_outputs_strong_forrefine = pseudo_outputs_strong.detach().clone()
            pseudo_outputs_strong_numpy = pseudo_outputs_strong_forrefine.cpu().numpy()
            pseudo_outputs_strong_color = pl_strong_embed(color_map, pseudo_outputs_strong_numpy)

            # (a) 3. encode gt labels (only for labeled data), replace the label of unlabeled data with weak pseudo labels
            label_batch_numpy = label_batch[:][: args.labeled_bs].cpu().numpy()
            label_batch_color = label_embed(color_map, label_batch_numpy)
            label_batch_color = torch.cat((label_batch_color.cuda(), pseudo_outputs_color[args.labeled_bs :].cuda()), dim=0) # lab+unlab
                
            loss = sup_loss + consistency_weight * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ############################################################################################################
            # (b) Latent Context Refinement Module
            # (b) 1. Weak Pseudo Label -> GT Label refinement
            t = dice_loss(pseudo_outputs_for_refine[:  args.labeled_bs].unsqueeze(1), label_batch[: args.labeled_bs].unsqueeze(1), oh_input=True)
            t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999
            # condition: semantic weak pl. input: semantic gt label
            lat_loss_sup, ref_outputs = refine_model(pseudo_outputs_color.cuda(), t, weak_batch.cuda(), training=True, good=label_batch_color.cuda())
            ref_outputs_soft = torch.softmax(ref_outputs, dim=1)

            sup_loss_cedice = ce_loss(ref_outputs[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                ref_outputs_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )
            sup_loss_ref = sup_loss_cedice + lat_loss_sup

            # The supervision for strong pseudo labels is the weak pseudo labels 
            # generated from the refine model in (b) 1. Another choice is to use the
            # weak pseudo labels generated from the segmentation model:
            # ref_pseudo_outputs = # pseudo_outputs_for_refine
            ref_soft = ref_outputs_soft
            ref_pseudo_mask = (normalize(ref_soft) > args.conf_thresh).float()
            ref_outputs_masked = ref_soft * ref_pseudo_mask
            ref_pseudo_outputs = torch.argmax(ref_outputs_masked.detach(), dim=1, keepdim=False) # lab+unlab

            # (b) 2. Strong Pseudo Label -> Weak Pseudo Label refinement
            t2 = dice_loss(pseudo_outputs_strong_forrefine[args.labeled_bs :].unsqueeze(1), ref_pseudo_outputs[args.labeled_bs:].unsqueeze(1), oh_input=True)
            t2 = torch.ones((pseudo_outputs_strong_color.shape[0]), dtype=torch.float32, device='cuda') * t2 * 999
            # condition: semantic strong pl. input: semantic weak pl.
            lat_loss_unsup, ref_outputs_strong = refine_model(pseudo_outputs_strong_color.cuda(), t2, strong_batch.cuda(), training=True, good=pseudo_outputs_color.cuda())
            ref_outputs_strong_soft = torch.softmax(ref_outputs_strong, dim=1) # lab+unlab

            ref_comp_loss, ref_as_weight = get_comp_loss(weak=ref_soft, strong=ref_outputs_strong_soft)
            unsup_loss_cedice = (
                ce_loss(ref_outputs_strong[args.labeled_bs :], ref_pseudo_outputs[args.labeled_bs :])
                + dice_loss(ref_outputs_strong_soft[args.labeled_bs :], ref_pseudo_outputs[args.labeled_bs :].unsqueeze(1))
                + ref_as_weight * ref_comp_loss
            ) 
            unsup_loss_ref = unsup_loss_cedice + lat_loss_unsup

            # ref_outputs_soft_for_refine = ref_outputs_soft.detach().clone() # lab+unlab
            ref_consistency_weight = consistency_weight if args.ref_consistency_weight == -1 else args.ref_consistency_weight
            refine_loss = sup_loss_ref + ref_consistency_weight * unsup_loss_ref
            refine_optimizer.zero_grad()
            refine_loss.backward()
            refine_optimizer.step()

            ############################ 
            # (c) Rectification loss for segmentation model
            if iter_num > args.refine_start:
                # compute t again, maybe not necessary
                t = dice_loss(pseudo_outputs_for_refine[:  args.labeled_bs].unsqueeze(1), label_batch[: args.labeled_bs].unsqueeze(1), oh_input=True)
                t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999
                # condition: semantic weak PL. input: pure noise
                ref_outputs = refine_model(pseudo_outputs_color.cuda(), t, weak_batch.cuda(), training=False)

                ref_outputs_soft_for_refine = torch.softmax(ref_outputs, dim=1)
                pseudo_mask = (normalize(ref_outputs_soft_for_refine) > args.conf_thresh).float()
                ref_outputs_soft_masked = ref_outputs_soft_for_refine * pseudo_mask
                pseudo_outputs_ref = torch.argmax(ref_outputs_soft_masked.detach(), dim=1, keepdim=False)
                
                # rectification loss, forward again for segmentation model as computation graph has been freed
                outputs_weak = model(weak_batch)
                outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
                unsup_label_rect_loss = ce_loss(outputs_weak[args.labeled_bs :], pseudo_outputs_ref[args.labeled_bs :]) + dice_loss(
                    outputs_weak_soft[args.labeled_bs :],
                    pseudo_outputs_ref[args.labeled_bs :].unsqueeze(1),
                )

                optimizer.zero_grad()
                unsup_label_rect_loss.backward()
                optimizer.step()

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            logging.info("iteration %d : mloss : %f, refsupce: %f, refsuplat: %f, refunsupce: %f, refunsuplat: %f, t: %f, t2: %f" % 
                         (iter_num, loss.item(), sup_loss_cedice.item(), lat_loss_sup.item(), unsup_loss_cedice.item(), lat_loss_unsup.item(), torch.mean(t).item(), torch.mean(t2).item()))

            if iter_num % 200 == 0:
                model.eval()
                refine_model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_jaccard = np.mean(metric_list, axis=0)[2]

                if performance > best_performance:
                    best_performance = performance
                    logging.info("BEST PERFORMANCE UPDATED AT ITERATION %d: Dice: %f, HD95: %f" % (iter_num, performance, mean_hd95))
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
                    # util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)

                for class_i in range(num_classes - 1):
                    logging.info(
                        "iteration %d: model_val_%d_dice : %f model_val_%d_hd95 : %f model_val_%d_jaccard : %f"
                        % (iter_num, class_i + 1, metric_list[class_i, 0], class_i + 1, metric_list[class_i, 1], class_i + 1, metric_list[class_i, 2])
                    )
                logging.info(
                    "iteration %d : model_mean_dice : %f model_mean_hd95 : %f model_mean_jaccard : %f"
                    % (iter_num, performance, mean_hd95, mean_jaccard)
                )
                
                ###############
                # TEST, only use the result of the best val model
                # test_func(num_classes, db_test, model, refine_model, iter_num, testloader)
                ########################################################

                model.train()
                refine_model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

def pl_weak_embed(color_map, pseudo_outputs_numpy):
    pseudo_outputs_color = torch.zeros((pseudo_outputs_numpy.shape[0], 3, pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2]), dtype=torch.float32)
    for i in range(pseudo_outputs_numpy.shape[0]):
        # Map each class value to a color value using the color map
        color_data = np.zeros((pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_numpy[i] == class_id] = color # color_data is a 2D array of RGB values, shape: (height, width, 3)
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_color[i] = color_tensor
    return pseudo_outputs_color

def pl_strong_embed(color_map, pseudo_outputs_strong_numpy):
    pseudo_outputs_strong_color = torch.zeros((pseudo_outputs_strong_numpy.shape[0], 3, pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2]), dtype=torch.float32)
    for i in range(pseudo_outputs_strong_numpy.shape[0]):
        color_data = np.zeros((pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_strong_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_strong_color[i] = color_tensor
    return pseudo_outputs_strong_color

def label_embed(color_map, label_batch_numpy):
    label_batch_color = torch.zeros((label_batch_numpy.shape[0], 3, label_batch_numpy.shape[1], label_batch_numpy.shape[2]), dtype=torch.float32, device='cuda')
    for i in range(label_batch_numpy.shape[0]):
        color_data = np.zeros((label_batch_numpy.shape[1], label_batch_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[label_batch_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        label_batch_color[i] = color_tensor
    return label_batch_color

def test_func(num_classes, db_test, model, refine_model, iter_num, testloader):
    metric_list_test = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model,
                        classes=num_classes,
                    )
        metric_list_test += np.array(metric_i)
    metric_list_test = metric_list_test / len(db_test)
    performance = np.mean(metric_list_test, axis=0)[0]
    mean_hd95 = np.mean(metric_list_test, axis=0)[1]
    mean_jaccard = np.mean(metric_list_test, axis=0)[2]

    for class_i in range(num_classes - 1):
        logging.info(
                        "(Test) iteration %d: model_val_%d_dice : %f model_val_%d_hd95 : %f model_val_%d_jaccard : %f"
                        % (iter_num, class_i + 1, metric_list_test[class_i, 0], class_i + 1, metric_list_test[class_i, 1], class_i + 1, metric_list_test[class_i, 2])
                    )
    logging.info(
                    "(Test) iteration %d : model_mean_dice : %f model_mean_hd95 : %f model_mean_jaccard : %f"
                    % (iter_num, performance, mean_hd95, mean_jaccard)
                )
    # writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./logs/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
        # shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))
    print(snapshot_path + "/log.log")
    logging.getLogger('').handlers = []
    logging.basicConfig(
        filename=snapshot_path + "/log.log",
        level=logging.DEBUG,
        filemode="w", 
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # create the log file
    # logging.basicConfig(filename=snapshot_path + "/log.log", filemode="w", format="%(name)s -> %(levelname)s: %(message)s", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if "brats" in args.root_path.lower():
        args.patch_size = [128, 128]
    logging.info(str(args))

    train(args, snapshot_path)