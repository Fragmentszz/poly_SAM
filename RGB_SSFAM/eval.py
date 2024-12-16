
from monai.metrics import DiceMetric, MeanIoU, SurfaceDiceMetric, SSIMMetric, GeneralizedDiceScore
from logging import log

import logging
# dice = DiceMetric()
# gd =  GeneralizedDiceScore()
# iou = MeanIoU()


import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from data import test_dataset
from SAM_Model.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=1024, help='training dataset size')
parser.add_argument('--data_dir', default=' ', help='test dataset path')
parser.add_argument('--model_type', type=str, default='vit_l', help='weight for edge loss')
parser.add_argument('--checkpoint', type=str,  default=' ', help='test from checkpoints')
parser.add_argument('--save_dir', default='./test_maps/', help='path where to save predicted maps')
parser.add_argument('--dataset', default='overall', help='test dataset')
opt = parser.parse_args()

model = Model(opt)
model.load_state_dict(torch.load(opt.checkpoint))
for param in model.parameters():
    param.requires_grad_(False)
model.cuda()
model.eval()

logging.basicConfig(filename='./eval_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
def test(test_loader, save_path):
    with torch.no_grad():
        batch_dice = []
        batch_gd = []
        batch_iou = []

        


        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            
            dice = DiceMetric()
            gd =  GeneralizedDiceScore()
            iou = MeanIoU()
            
            image = image.cuda()


            res, _ = model(image, None)

            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = torch.tensor(res).reshape(1,1,res.shape[0],res.shape[1])
            gt = torch.tensor(gt).reshape(1,1,gt.shape[0],gt.shape[1])
            dice(res, gt)
            gd(res, gt)
            iou(res, gt)
            final_dice = dice.aggregate().numpy()[0]
            final_gd = gd.aggregate().numpy()[0]
            final_iou = iou.aggregate().numpy()[0]
            batch_dice.append(final_dice)
            batch_gd.append(final_gd)
            batch_iou.append(final_iou)
            print('save img to: ', save_path + '/'+ name)
            res = res.squeeze().cpu().numpy()
            res = np.round(res * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, name), res)
        logging.info(f'Mean val dice: {sum(batch_dice) / len(batch_dice)}')
        logging.info(f'Mean val gd: {sum(batch_gd) / len(batch_gd)}')
        logging.info(f'Mean val iou: {sum(batch_iou) / len(batch_iou)}')
        
        print(f'Mean val dice: {sum(batch_dice) / len(batch_dice)}')
        print(f'Mean val gd: {sum(batch_gd) / len(batch_gd)}')
        print(f'Mean val iou: {sum(batch_iou) / len(batch_iou)}')
    print('Test Done!')
if __name__ == '__main__':
    
    if opt.dataset == 'overall':
        image_root = os.path.join(opt.data_dir, 'images') + '/'
        gt_root = os.path.join(opt.data_dir, 'gts') + '/'
        test_loader = test_dataset(image_root, gt_root, opt.img_size)
        save_path = opt.save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        test(test_loader, save_path)
        
    elif opt.dataset == 'divide':
        # CVC-300  CVC-ClinicDB  CVC-ColonDB  ETIS-LaribPolypDB  Kvasir
        test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
        for dataset in test_datasets:
            image_root = os.path.join(opt.data_dir, dataset, 'images') + '/'
            gt_root = os.path.join(opt.data_dir, dataset, 'gts') + '/'
            test_loader = test_dataset(image_root, gt_root, opt.img_size)
            save_path = os.path.join(opt.save_dir, dataset)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"========{dataset} test=====")
            logging.info(f"========{dataset} test=====")
            test(test_loader, save_path)

    else:
        dataset = opt.dataset
        save_path = os.path.join(opt.save_dir, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = os.path.join(opt.data_dir, dataset, 'images') + '/'
        gt_root = os.path.join(opt.data_dir, dataset, 'gts') + '/'
        test_loader = test_dataset(image_root, gt_root, opt.img_size)

        test(test_loader, save_path)