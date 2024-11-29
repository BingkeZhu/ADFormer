import os
import random
import argparse
import time
import math
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.funcs import rot_img, translation_img, hflip_img, norm_img
from sklearn.metrics import roc_auc_score
from model import ADformer
from utils.optimizer import build_optimizer

channels = 512
tokens = 28*28
feature_size = 28


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--obj', type=str, default='bottle')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='./data/mvtec_anomaly_detection/')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    parser.add_argument('--comment', type=str, default='test',help='comment')
    args = parser.parse_args()
    args.input_channel = 3

    # set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # logging path
    args.prefix = time_file_str()
    args.save_dir = './logs_mvtec/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_model_dir = './logs_mvtec/' + args.comment + '/' + str(args.shot) + '/' + args.obj + '/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    log = open(os.path.join(args.save_dir, 'log_{}_{}_{}.txt'.format(str(args.shot),args.obj,args.comment)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # creating model
    model = ADformer().to(device)

    # optimizer
    class Model_args(object):
        def __init__(self) -> None:
            self.weight_decay_norm = 0
            self.weight_decay_embed = 0
            self.lr = args.lr
            self.weight_decay = 1e-4
            self.backbone_lr_scale = 0
            self.optimizer = "ADAMW"
            self.momentum = 0.9

    optimizer_cfg = Model_args()
    optimizer = build_optimizer(optimizer_cfg, model)
    init_lrs = args.lr

    # dataset
    print('Loading Datasets') 
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    train_dataset = FSAD_Dataset_train(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size, shot=1, batch=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # checkpoint path
    save_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.comment))
    start_time = time.time()
    epoch_time = AverageMeter()
    img_roc_auc_old = 0.0
    per_pixel_rocauc_old = 0.0

    # loading Support Set
    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_10.pt')
    print_log((f'---------{args.comment}--------'), log)

    # epoch
    for epoch in range(1, args.epochs + 1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        # test epoch
        if epoch <= args.epochs:
            image_auc_list = []
            pixel_auc_list = []
            for inference_round in tqdm(range(args.inferences)):
                scores_list, test_imgs, gt_list, gt_mask_list = test(model, inference_round, fixed_fewshot_list,test_loader)
                scores = np.asarray(scores_list)

                # Normalize
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

                # Calculate the image-level ROC AUC score.
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)
                
                img_roc_auc = roc_auc_score(gt_list, img_scores)
                image_auc_list.append(img_roc_auc)

                # Calculate the pixel-level ROC AUC score.
                gt_mask = np.asarray(gt_mask_list)
                gt_mask = (gt_mask > 0.5).astype(np.int_)
                
                per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
                pixel_auc_list.append(per_pixel_rocauc)
            
            image_auc_list = np.array(image_auc_list)
            pixel_auc_list = np.array(pixel_auc_list)
            mean_img_auc = np.mean(image_auc_list, axis = 0)
            mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)

            print('Img-level AUC:',mean_img_auc)
            print('Pixel-level AUC:', mean_pixel_auc)

            if mean_img_auc + mean_pixel_auc > img_roc_auc_old + per_pixel_rocauc_old:
                state = model.state_dict()
                torch.save(state, save_name)
                per_pixel_rocauc_old = mean_pixel_auc
                img_roc_auc_old = mean_img_auc
            # early stop
            if mean_img_auc + mean_pixel_auc + 0.05 < img_roc_auc_old + per_pixel_rocauc_old:
                return

            print_log(('Test Epoch(img, pixel): {} ({:.6f}, {:.6f}) best: ({:.3f}, {:.3f})'
            .format(epoch-1, mean_img_auc, mean_pixel_auc, img_roc_auc_old, per_pixel_rocauc_old)), log)


        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        # training
        train(model, epoch, train_loader, optimizer, log)
        # adjust lr
        adjust_learning_rate(optimizer, init_lrs, epoch, args)
        
        # shuffling training data
        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        
    log.close()

def train(model, epoch, train_loader, optimizer, log):

    model.train()
    model.backbone.eval()

    total_loss = AverageMeter()

    for (query_img, support_img_list, _) in tqdm(train_loader):

        optimizer.zero_grad()

        query_img = query_img.squeeze(0).to(device)

        support_img = support_img_list.squeeze(0).to(device)
        B,K,C,H,W = support_img.shape
        support_img = support_img.view(B * K, C, H, W)

        query_transform_img = torchvision.transforms.RandomRotation(degrees=(0,90))(query_img)
        
        # [B*K, C, H*W]
        #with torch.no_grad():
        support_feat = model(support_img).reshape([-1,tokens,channels])
        query_transform_feat = model(query_transform_img).reshape([-1,tokens,channels])
        support_feat = torch.cat([support_feat,query_transform_feat],dim=1)

        query_feat = model(query_img).reshape([-1,tokens,channels])

        query_feat = F.normalize(query_feat, dim=-1)
        support_feat = F.normalize(support_feat, dim=-1).transpose(-2,-1).contiguous()

        # Calculate the similarity between the query and each patch of the support 
        # (the support includes the geometric transformation of the query and another randomly selected picture from the dataset).
        # [batch_size, H*W, H*W*2]
        sim = torch.matmul(query_feat, support_feat)

        # Calculate the closest match in the support for each patch of the query.
        sim_match = torch.max(sim,dim=-1).values.transpose(0,1)

        # The maximum similarity value of the matching results (positive examples).
        match_max = torch.max(sim_match,dim=-1).values
        # The minimum similarity value of the matching results (negative examples).
        match_min = torch.min(sim_match,dim=-1).values

        # triplet loss
        loss = (match_min - match_max + 1).mean()

        # backward
        loss.backward()
        total_loss.update(loss.item(),B)

        optimizer.step()

    print_log(('Train Epoch: {} Loss: {:.6f}'.format(epoch, total_loss.avg)), log)

def test(model, cur_epoch, fixed_fewshot_list, test_loader):

    model.eval()

    support_img = fixed_fewshot_list[cur_epoch]
    support_img = norm_img(support_img)

    augment_support_img = support_img
    
    # Perform data augmentation on the support image.
    # rotate
    for angle in [-np.pi * 7 / 8, -np.pi * 3 / 4, -np.pi * 5 / 8, -np.pi * 3 / 8, -np.pi / 4, -np.pi / 8, 
    np.pi / 8, np.pi / 4, np.pi * 7 / 8, np.pi * 3 / 4, np.pi * 5 / 8, np.pi * 3 / 8, np.pi / 2, -np.pi / 2, np.pi]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)

    # translation
    for a, b in [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),(0.1, -0.1),
                (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2),(0.2, -0.2)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)

    # flip
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)

    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    
    # Calculate the features of the support.
    with torch.no_grad():
        support_feat = model(augment_support_img.to(device))

    support_feat = support_feat.reshape(-1, channels)

    query_imgs = []
    gt_list = []
    mask_list = []
    diff_list = []
    
    for (query_img, _, mask, y) in test_loader:
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())
        
        # Calculate the features of the query.
        with torch.no_grad():
            query_feat = model(query_img.to(device))

        query_feat = query_feat.reshape(tokens, channels)
        
        # Calculate the similarity between each patch of the query and each patch of the support image.
        # The smaller the similarity is, the more likely it indicates an anomaly. 
        # Therefore, the negative value is taken as the anomaly score.
        diff = - torch.matmul(F.normalize(query_feat, dim=-1), F.normalize(support_feat, dim=-1).t().contiguous())

        # Each patch of the query is matched with the most similar patch in the support.
        diff = torch.min(diff,dim=-1).values

        # Upsample the anomaly scores to 224 * 224.
        diff = diff.reshape(1,1,feature_size,feature_size)
        diff = torch.nn.Upsample(size=(224, 224), mode='bilinear')(diff)
        diff = diff.reshape(224,224)

        diff_list.append(diff.detach().cpu().numpy())

    return diff_list, query_imgs, gt_list, mask_list


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr  * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    print(cur_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == '__main__':
    main()
