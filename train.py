import os
import random
import argparse
import time
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.funcs import  rot_img, translation_img, norm_img
from sklearn.metrics import roc_auc_score
from model import Backbone, ADformer, hungarian_matching
from utils.optimizer import build_optimizer

channels = 512
tokens = 28*28
feature_size = 28


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Transformer')
    parser.add_argument('--obj', type=str, default='bottle')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='../data/mvtec/')
    parser.add_argument('--epochs', type=int, default=20, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate of others in AdamW')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    parser.add_argument('--comment', type=str, default='default',help='comment')
    args = parser.parse_args()
    args.input_channel = 3
    
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Set log save path
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


    # Create model
    model = ADformer().to(device)
    backbone = Backbone().to(device)

    
    # Create optimizer parameters
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

    # Load dataset
    print('Loading Datasets') 
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    train_dataset = FSAD_Dataset_train(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size, shot=1, batch=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # Set checkpoint save path
    save_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.comment))
    start_time = time.time()
    epoch_time = AverageMeter()
    img_roc_auc_old = 0.0
    per_pixel_rocauc_old = 0.0

    # Load Support Set
    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_10.pt')
    print_log((f'---------{args.comment}--------'), log)

    # epochs
    for epoch in range(1, args.epochs + 1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        # Test epoch
        if epoch <= args.epochs:
            image_auc_list = []
            pixel_auc_list = []
            for inference_round in tqdm(range(args.inferences)[:1]):
                scores_list, test_imgs, gt_list, gt_mask_list = test(model, inference_round, fixed_fewshot_list,
                                                                     test_loader, backbone)
                scores = np.asarray(scores_list)

                # Normalization
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
                scores = np.nan_to_num(scores)

                # Calculate image-level ROC AUC score
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)
                
                img_roc_auc = roc_auc_score(gt_list, img_scores)
                image_auc_list.append(img_roc_auc)

                # Calculate pixel-level ROC AUC score
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

            # Save model parameters
            if mean_img_auc + mean_pixel_auc > img_roc_auc_old + per_pixel_rocauc_old:
                state = model.state_dict()
                torch.save(state, save_name)
                per_pixel_rocauc_old = mean_pixel_auc
                img_roc_auc_old = mean_img_auc

            print_log(('Test Epoch(img, pixel): {} ({:.6f}, {:.6f}) best: ({:.3f}, {:.3f})'
            .format(epoch-1, mean_img_auc, mean_pixel_auc, img_roc_auc_old, per_pixel_rocauc_old)), log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        # Training
        train(model, epoch, train_loader, optimizer, log, backbone)
        # Adjust learning rate
        adjust_learning_rate(optimizer, init_lrs, epoch, args)
        
        # Shuffle training data
        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        
    log.close()

def train(model, epoch, train_loader, optimizer, log, backbone):

    model.train()
    backbone.eval()
    total_loss = AverageMeter()

    if epoch % 2 == 1: # Consistency-Enhanced Loss
        for (query_img, support_img_list, _) in tqdm(train_loader):

            optimizer.zero_grad()

            # Random rotation degree
            degree = random.randint(1,360)

            query_img = query_img.squeeze(0)

            # Rotate image
            query_img_rotate = rot_img(query_img, degree*np.pi/180)

            query_img = query_img.to(device)
            query_img_rotate = query_img_rotate.to(device)

            B,C,H,W = query_img.shape

            # Pass both images through backbone
            query_feat = backbone(query_img)
            query_feat_bn = query_feat
            query_feat_rotate = backbone(query_img_rotate).detach().cpu().transpose(-2,-1).reshape([B,channels,28,28])
            
            # Apply inverse transformation to the rotated feature
            query_feat_rotate = rot_img(query_feat_rotate, -degree*np.pi/180).to(device).reshape([B,channels,28*28]).transpose(-2,-1)

            # Input to Transformer
            query_feat = model(query_feat)
            query_feat_rotate = model(query_feat_rotate).detach()

            # Normalize features
            query_feat = F.normalize(query_feat, dim=-1)
            query_feat_bn = F.normalize(query_feat_bn, dim=-1)
            query_feat_rotate = F.normalize(query_feat_rotate, dim=-1).transpose(-2,-1).contiguous()

            # Similarity
            sim_self = torch.matmul(query_feat_bn,query_feat_bn.transpose(-2,-1).contiguous())
            sim_rot = torch.matmul(query_feat,query_feat_rotate)

            loss = nn.MSELoss()(sim_self, sim_rot)

            loss.backward()
            total_loss.update(loss.item(),B)
            optimizer.step()


    else: #L_triplet
        for (query_img, support_img_list, _) in tqdm(train_loader):

            optimizer.zero_grad()

            query_img = query_img.squeeze(0).to(device)


            support_img = support_img_list.squeeze(0)
            B,K,C,H,W = support_img.shape
            support_img = support_img.view(B * K, C, H, W).to(device)


            query_feat_bn = backbone(query_img)
            support_feat_bn = backbone(support_img).detach()


            query_feat = model(query_feat_bn)
            support_feat = model(support_feat_bn).detach()


            query_feat = F.normalize(query_feat, dim=-1)
            support_feat = F.normalize(support_feat, dim=-1).transpose(-2,-1).contiguous()


            sim = torch.matmul(query_feat, support_feat)
            sim_match = hungarian_matching(sim)

            # The maximum similarity value of the matching results (positive examples).
            match_max = torch.max(sim_match,dim=-1).values
            # The minimum similarity value of the matching results (negative examples).
            match_min = torch.min(sim_match,dim=-1).values

            # Triplet Loss
            loss = (match_min - match_max + 1).mean()


            loss.backward()
            total_loss.update(loss.item(),B)
            optimizer.step()


    print_log(('Train Epoch: {} Loss: {:.6f}'.format(epoch, total_loss.avg)), log)

def test(model, cur_epoch, fixed_fewshot_list, test_loader, backbone):

    model.eval()
    backbone.eval()

    support_img = fixed_fewshot_list[cur_epoch]
    support_img = norm_img(support_img)

    augment_support_img = support_img
    
    # rotate img with small angle
    for angle in [-np.pi * 7 / 8, -np.pi * 3 / 4, -np.pi * 5 / 8, -np.pi * 3 / 8, -np.pi / 4, -np.pi / 8, 
    np.pi / 8, np.pi / 4, np.pi * 7 / 8, np.pi * 3 / 4, np.pi * 5 / 8, np.pi * 3 / 8, np.pi / 2, -np.pi / 2, np.pi]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    # translate img
    for a, b in [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),(0.1, -0.1),
                (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2),(0.2, -0.2)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)

    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    

    with torch.no_grad():
        support_feat = backbone(augment_support_img.to(device))
        support_feat = model(support_feat)
        
    support_feat = support_feat.reshape(-1, channels)

    
    query_imgs = []
    gt_list = []
    mask_list = []
    diff_list = []
    
    for (query_img, _, mask, y) in test_loader:
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())

        with torch.no_grad():
            query_feat = backbone(query_img.to(device))
            query_feat = model(query_feat)

        query_feat = query_feat.reshape(tokens, channels)

      
        sim = torch.matmul(F.normalize(query_feat, dim=-1), F.normalize(support_feat, dim=-1).t().contiguous()).unsqueeze(0)

        sim_max = hungarian_matching(sim)
        diff = 1 - sim_max

        diff = diff.reshape(1,1,feature_size,feature_size)
        diff = torch.nn.Upsample(size=(224, 224), mode='bilinear')(diff)
        diff = diff.reshape(224,224)
        diff_list.append(diff.detach().cpu().numpy())

    return diff_list, query_imgs, gt_list, mask_list



def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr  * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    print(cur_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == '__main__':
    main()
