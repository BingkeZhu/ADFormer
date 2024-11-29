import os
import random
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from mvtec import  FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.funcs import  rot_img, translation_img, hflip_img, norm_img
from sklearn.metrics import roc_auc_score
from model import ADformer, Backbone, hungarian_matching


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

channels = 512
tokens = 28*28
feature_size = 28

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--obj', type=str, default='bottle')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='../data/mvtec/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    parser.add_argument('--comment', type=str, default='default',help='comment')
    args = parser.parse_args()
    args.input_channel = 3
    args.epochs = 1

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

    log = open(os.path.join(args.save_dir, 'log_{}_{}_{}_test.txt'.format(str(args.shot),args.obj,args.comment)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # creating model and loading weights
    model = ADformer().to(device)
    checkpoint_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.comment))
    checkpoint = torch.load(checkpoint_name,map_location="cuda:0")
    model.load_state_dict(checkpoint)

    backbone = Backbone().to(device)

    # loading dataset
    print('Loading Datasets') 
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # loading Support Set
    start_time = time.time()
    epoch_time = AverageMeter()
    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_10.pt')
    print_log((f'---------{args.comment}--------'), log)

    # test epoch
    for epoch in range(1, args.epochs + 1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

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

                # Calculate the image-level ROC AUC score.
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)
                for i in range(len(gt_list)):
                    print(i, gt_list[i], img_scores[i])
                
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

            print_log(('Test Epoch(img, pixel): {} ({:.6f}, {:.6f})'
            .format(epoch-1, mean_img_auc, mean_pixel_auc)), log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        
    log.close()

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

if __name__ == '__main__':
    main()


