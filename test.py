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
from model import ADformer
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:3' if use_cuda else 'cpu')
pic_to_show = 50
show_pic = True

channels = 512
tokens = 28*28
feature_size = 28

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--obj', type=str, default='bottle')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='./data/mvtec_anomaly_detection/')
    parser.add_argument('--img_size', type=int, default=224)
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

    log = open(os.path.join(args.save_dir, 'log_{}_{}_{}_test.txt'.format(str(args.shot),args.obj,args.comment)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # creating model and loading weights
    model = ADformer().to(device)
    checkpoint_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.comment))
    checkpoint = torch.load(checkpoint_name,map_location="cuda:3")
    model.load_state_dict(checkpoint)

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
            for inference_round in range(args.inferences):
                scores_list, test_imgs, gt_list, gt_mask_list = test(model, inference_round, fixed_fewshot_list,
                                                                     test_loader)
                scores = np.asarray(scores_list)

                # Normalization
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

                scores = np.nan_to_num(scores)

                # Visualize the anomaly scores.
                if show_pic:
                    test_imgs[pic_to_show] = test_imgs[pic_to_show] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
                    plt.imshow(np.transpose(test_imgs[pic_to_show], (1, 2, 0)))
                    plt.savefig('test_pic.jpg')

                    plt.imshow(np.transpose(test_imgs[pic_to_show], (1, 2, 0)))
                    plt.imshow(scores[pic_to_show],alpha=0.5)
                    plt.colorbar()
                    plt.savefig('test_pic_score.jpg')

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

    # Save all the patches in the support for piecing together the matching results later.
    support_img_show = np.asarray(augment_support_img)
    support_img_show = np.transpose(support_img_show,(0,2,3,1))
    support_image_patches = []
    for i in range(support_img_show.shape[0]):
        patches = patchify(support_img_show[i], (8, 8, 3), step=8)
        patches = patches.reshape(28*28,8,8,3)
        support_image_patches.append(patches)
    support_image_patches = np.array(support_image_patches)
    support_image_patches = support_image_patches.reshape(-1,8,8,3) 
    
    # Calculate the features of the support.
    with torch.no_grad():
        support_feat = model(augment_support_img.to(device))
    support_feat = support_feat.reshape(-1, channels)

    query_imgs = []
    gt_list = []
    mask_list = []
    diff_list = []
    
    i = 0
    for (query_img, _, mask, y) in test_loader:
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())

        
        with torch.no_grad():
            query_feat = model(query_img.to(device))
            

        query_feat = query_feat.reshape(tokens, channels)
        
        # Calculate the similarity between each patch of the query and each patch of the support image.
        # The smaller the similarity is, the more likely it indicates an anomaly. 
        # Therefore, the negative value is taken as the anomaly score.
        matrix = - torch.matmul(F.normalize(query_feat, dim=-1), F.normalize(support_feat, dim=-1).t().contiguous())
        
        # Each patch of the query is matched with the most similar patch in the support.
        diff = torch.min(matrix,dim=-1).values

        if i == pic_to_show:
            print(diff)

        # Upsample the anomaly scores to 224 * 224.
        diff = diff.reshape(1,1,feature_size,feature_size)
        diff = torch.nn.Upsample(size=(224, 224), mode='bilinear')(diff)
        diff = diff.reshape(224,224)

        # Output the stitched image according to the matching results.
        if show_pic and i == pic_to_show:
            support_image_patches = support_image_patches[torch.min(matrix,dim=-1).indices.cpu().flatten()].reshape(28,28,1,8,8,3)
            support_pic = unpatchify(support_image_patches,(224,224,3))
            support_pic = support_pic * np.array([0.229, 0.224, 0.225]).reshape(1,1,3) + np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
            plt.imshow(support_pic)
            plt.savefig('support_pic.jpg')
        i += 1

        diff_list.append(diff.detach().cpu().numpy())

    return diff_list, query_imgs, gt_list, mask_list


if __name__ == '__main__':
    main()


