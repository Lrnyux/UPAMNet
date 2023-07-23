import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import logging
import os
import time
from tqdm import tqdm
import random
from utils import AverageMeter,ConsoleLogger,calc_rmse,calc_psnr,calc_ssim,save_results,InputPadder
from model import UPAMNet
from loss import *
import lpips


def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        tar.add(root, arcname='code', recursive=True)

def get_args():
    parser = argparse.ArgumentParser()

    # =========for hyper parameters===
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--trial',type=str, default='baseline_MSE')
    parser.add_argument('--mode', type=str, default='train',choices=['train','test','test_offline'])
    parser.add_argument('--seed',type=int,default=1)

    # ==========define the task==============
    parser.add_argument('--task',type=str,default='SR',choices=['SR','Denoise'])
    parser.add_argument('--dataset_type', type=str, default='syn', choices=['syn'])



    # =========for SR dataset============
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--fill_type',type=str,default='bic', choices=['bic', 'empty'])
    parser.add_argument('--train_data_root', type=str,
                        default='')
    parser.add_argument('--val_data_root', type=str, default='')
    parser.add_argument('--test_data_root', type=str, default='')

    parser.add_argument('--train_data_meta_root', type=str,
                        default='')
    parser.add_argument('--val_data_meta_root', type=str, default='')
    parser.add_argument('--test_data_meta_root', type=str, default='')
    # =========for training===========

    parser.add_argument('--train_batchsize', type=int, default=32)
    parser.add_argument('--val_batchsize', type=int, default=16)
    parser.add_argument('--max_epoch', default=20, help='max training epoch', type=int)

    parser.add_argument('--lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--weight_decay', default=0.0001, help='decay of learning rate', type=float)

    parser.add_argument('--freq_print_train', default=20, help='Printing frequency for training', type=int)
    parser.add_argument('--freq_print_val', default=20, help='Printing frequency for validation', type=int)
    parser.add_argument('--freq_print_test', default=50, help='Printing frequency for test', type=int)
    parser.add_argument('--load_model', type=str, default='')

    # ==========loss function============
    parser.add_argument('--loss_type',type=str,default='percept_patch_prior',choices=['mse','percept','percept_patch','percept_patch_prior'])
    parser.add_argument('--w_per',type=float,default=5e-3)
    parser.add_argument('--w_back_per',type=float,default=0.1)
    parser.add_argument('--w_object_per',type=float,default=1.0)
    parser.add_argument('--w_edge_per',type=float,default=2.0)

    # ========for model ==============
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--inner_channel', type=int, default=16)
    parser.add_argument('--norm_groups', type=int, default=16)
    parser.add_argument('--res_blocks',type=int, default=2)
    parser.add_argument('--channel_mults',type=list,default=[1,2,4,8])

    parser.add_argument('--attn_res', type=list, default=[16,32,64])

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--ifres', type=bool, default=True)
    parser.add_argument('--save_gray', type=int, default=-1) # 0 for color images; 1 for gray images;
    parser.add_argument('--save_epoch',type=int, default=1)

    # parse configs
    args = parser.parse_args()
    return args



class PA_dataset(Dataset):

    #  baseline method for photoacoustic image super resolution
    #  input is the bicubic interpolation of the down sampling image

    def __init__(self,root_dir,meta_root_dir,args,stage='Train'):
        self.root_dir = root_dir
        self.meta_root_dir = meta_root_dir
        self.args = args
        self.scale = self.args.scale
        self.stage = stage
        self.filelist = []
        self.edge_filelist = []
        self.object_filelist = []
        root_list = sorted(os.listdir(self.root_dir))
        for idx in range(len(root_list)):
            per_root = os.path.join(self.root_dir,root_list[idx])
            self.filelist.append(per_root)
            edge_root = os.path.join(self.meta_root_dir,'edge',root_list[idx])
            self.edge_filelist.append(edge_root)
            object_root = os.path.join(self.meta_root_dir,'object',root_list[idx])
            self.object_filelist.append(object_root)

        self.length = len(self.filelist)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        s = self.scale
        file_root = self.filelist[index]
        image = np.array(Image.open(file_root))
        h, w = image.shape
        if self.args.fill_type == 'bic':
            down = np.array(Image.open(file_root).resize((w // s, h // s), Image.Resampling.BICUBIC))
            target = np.array(Image.open(file_root).resize((w // s, h // s), Image.Resampling.BICUBIC).resize((w , h ), Image.Resampling.BICUBIC))
        if self.args.fill_type == 'bilinear':
            down = np.array(Image.open(file_root).resize((w // s, h // s), Image.Resampling.BILINEAR))
            target = np.array(Image.open(file_root).resize((w // s, h // s), Image.Resampling.BILINEAR).resize((w , h ), Image.Resampling.BICUBIC))
        if self.args.fill_type == 'equal_inter':
            down = np.array(Image.open(file_root))[::s,::s]
            target = np.zeros((h,w))
            target[::s,::s] = down

        down = down.reshape(h//s,w//s,1)
        target = target.reshape(h,w,1)
        image = image.reshape(h,w,1)

        edge_file_root = self.edge_filelist[index]
        object_file_root = self.object_filelist[index]
        image_edge = np.array(Image.open(edge_file_root))
        image_object = np.array(Image.open(object_file_root))
        mask = np.zeros((h, w))
        mask[image_object == 255] = 1
        mask[image_edge == 255] = 2
        mask = mask.reshape(h, w, 1)
        # ==============augmentation===================================
        if self.stage == 'Train':
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5

            if hflip:
                image = image[:, ::-1, :]
                target = target[:,::-1,:]
                down = down[:,::-1,:]
                mask = mask[:,::-1,:]

            if vflip:
                image = image[::-1, :, :]
                target = target[::-1,:,:]
                down = down[::-1,:,:]
                mask = mask[::-1,:,:]

            if rot90:
                image = np.rot90(image, 1)
                target = np.rot90(target,1)
                down = np.rot90(down,1)
                mask = np.rot90(mask,1)

        #  perform normalization on image
        d_min = np.min(target)
        d_max = np.max(target)
        image = (image - d_min) / (d_max - d_min)
        target = (target - d_min) / (d_max - d_min)
        down = (down - d_min) / (d_max - d_min)


        image = np.transpose(image,[2,0,1])
        target = np.transpose(target,[2,0,1])
        down = np.transpose(down,[2,0,1])

        mask = np.transpose(mask, [2, 0, 1])

        sample={}
        sample['gt']=image.copy()
        sample['input'] = target.copy()
        sample['down'] = down.copy()
        sample['mask'] = mask.copy()
        sample['root'] = file_root
        sample['scale'] = self.scale
        sample['d_min'] = d_min
        sample['d_max'] = d_max
        return sample




def train():
    args = get_args()
    LOGGER = ConsoleLogger('train_'+args.trial, 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    torch.manual_seed(args.seed)
    # -------save current code---------------------------
    save_code_root = os.path.join(logdir, 'code.tar')
    dst_root = os.path.abspath(__file__)
    create_code_snapshot(dst_root, save_code_root)
    # dataset============================================================
    train_set = PA_dataset(args.train_data_root,args.train_data_meta_root,args,stage='Train')
    train_dataloader = DataLoader(train_set,batch_size=args.train_batchsize,shuffle=True,num_workers=16,drop_last=True)
    val_set = PA_dataset(args.val_data_root,args.val_data_meta_root, args, stage='Val')
    val_dataloader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16, drop_last=False)
    test_set = PA_dataset(args.test_data_root,args.test_data_meta_root, args, stage='Test')
    test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16, drop_last=False)
    LOGGER.info('Initial Dataset Finished')
    #====================================================================

    # model
    # model = unet()
    model = UPAMNet(inner_channel=args.inner_channel,norm_groups=args.norm_groups,channel_mults=args.channel_mults,attn_res=args.attn_res,res_blocks=args.res_blocks)
    device = torch.device(f'cuda:{args.gpu}')
    model = model.cuda(device)
    Loss_img =  eval(args.loss_type + '(args=args)').to(device)
    best_perf = 100.0

    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    step_size = int(args.max_epoch * 0.4 * len(train_dataloader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                     gamma=0.1)
    # ================train process=============================================
    for epoch in range(args.max_epoch):
        LOGGER.info(f'---------------Training epoch : {epoch}-----------------')
        batch_time = AverageMeter()
        loss_log = AverageMeter()
        loss_val_log = AverageMeter()
        loss_val_psnr = AverageMeter()
        loss_val_ssim = AverageMeter()
        start = time.time()

        model.train()
        for it, batch in enumerate(train_dataloader, 0):
            rgb = batch['input'].to(torch.float32).cuda(device)
            gt = batch['gt'].to(torch.float32).cuda(device)
            down = batch['down'].to(torch.float32).cuda(device)
            mask = batch['mask'].to(torch.float32).cuda(device)
            batch_size = len(rgb)

            output = model(rgb)
            loss_term = Loss_img(pred=output,gt=gt-rgb,epoch=epoch,mask=mask)


            if (args.loss_type == 'mse') :
                loss = loss_term['mse_loss']


            if (args.loss_type == 'percept') or (args.loss_type == 'percept_patch'):
                loss = loss_term['mse_loss'] + args.w_per * loss_term['per_loss']
                if it % args.freq_print_train == 0:
                    print('MSE-Term:{:.5f} Per-Term:{:.5f}'.format(loss_term['mse_loss'].item(),loss_term['per_loss'].item()))

            if (args.loss_type == 'percept_patch_prior') :
                loss = loss_term['mse_loss'] + args.w_per * (args.w_back_per * loss_term['per_back_loss'] + args.w_object_per * loss_term['per_object_loss'] + args.w_edge_per * loss_term['per_edge_loss'])
                if it % args.freq_print_train == 0:
                    print('MSE-Term:{:.5f} Per-Term-Back:{:.5f} Per-Term-Obj:{:.5f} Per-Term-Edge:{:.5f}'.format(loss_term['mse_loss'].item(),loss_term['per_back_loss'].item(),loss_term['per_object_loss'].item(),loss_term['per_edge_loss'].item()))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # ===============Logging the info and Save the model================
            batch_time.update(time.time() - start)
            loss_log.update(loss.detach(), batch_size)
            if it % args.freq_print_train == 0:
                message = 'Epoch : [{0}][{1}/{2}]  Learning rate  {learning_rate:.7f}\t' \
                          'Batch Time {batch_time.val:.3f}s ({batch_time.ave:.3f})\t' \
                          'Speed {speed:.1f} samples/s \t' \
                          'Loss_train {loss1.val:.5f} ({loss1.ave:.5f})\t'.format(
                    epoch, it, len(train_dataloader), learning_rate=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time, speed=batch_size / batch_time.val, loss1=loss_log)
                LOGGER.info(message)
            start = time.time()

        # ================validation process=============================================
        with torch.no_grad():
            model.eval()
            for it, batch in enumerate(val_dataloader, 0):
                rgb = batch['input'].to(torch.float32).cuda(device)
                gt = batch['gt'].to(torch.float32).cuda(device)
                down = batch['down'].to(torch.float32).cuda(device)
                batch_size = len(rgb)
                minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
                # =========feed into the network=========================
                output = rgb + model(rgb)
                # print(output.shape)
                for idx in range(len(output)):
                    rmse = calc_rmse(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                    loss_val_log.update(rmse, 1)
                    psnr = calc_psnr(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                    loss_val_psnr.update(psnr, 1)
                    ssim = calc_ssim(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                    loss_val_ssim.update(ssim, 1)
            message = 'RMSE_val {loss1.ave:.5f} || PSNR {loss2.ave:.5f} || SSIM {loss3.ave:.5f}\t'.format(loss1=loss_val_log,loss2=loss_val_psnr,loss3=loss_val_ssim)
            LOGGER.info(message)

        if best_perf > loss_val_log.ave:
            best_perf = loss_val_log.ave
            checkpoint_dir = os.path.join(logdir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
            states = dict()
            states['model_state_dict'] = model.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(states, os.path.join(checkpoint_dir, 'best_perf.tar'))

        if args.save_epoch == 1:
            checkpoint_dir = os.path.join(logdir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
            states = dict()
            states['model_state_dict'] = model.state_dict()
            states['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(states, os.path.join(checkpoint_dir, 'checkpoint_'+str(epoch)+'.tar'))

    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
    states = dict()
    states['model_state_dict'] = model.state_dict()
    states['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(states, os.path.join(checkpoint_dir, 'last.tar'))

    LOGGER.info('Finish Training')


    LOGGER.info('Start Testing the saved model')
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_perf.tar'))

    model.load_state_dict(checkpoint['model_state_dict'])
    LOGGER.info(f'---------------Finishing loading models----------------')

    rmse_log = AverageMeter()
    psnr_log = AverageMeter()
    ssim_log = AverageMeter()
    lpips_log = AverageMeter()
    loss_lpips = lpips.LPIPS(net='alex').cuda(device)

    with torch.no_grad():
        model.eval()
        for it, batch in enumerate(test_dataloader, 0):
            rgb = batch['input'].to(torch.float32).cuda(device)
            gt = batch['gt'].to(torch.float32).cuda(device)
            down = batch['down'].to(torch.float32).cuda(device)
            batch_size = len(rgb)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
            # =========feed into the network=========================
            output = rgb + model(rgb)
            # print(output.shape)
            for idx in range(len(output)):
                rmse = calc_rmse(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                rmse_log.update(rmse, 1)
                psnr = calc_psnr(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                psnr_log.update(psnr, 1)
                ssim = calc_ssim(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                ssim_log.update(ssim, 1)

            output_exp = output.expand(len(output),3,rgb.shape[2],rgb.shape[3])
            output_exp = torch.clip(output_exp,-1,1)
            gt_exp = gt.expand(len(gt),3,rgb.shape[2],rgb.shape[3])
            gt_exp = torch.clip(gt_exp,-1,1)
            lpips_error = loss_lpips(output_exp,gt_exp)
            lpips_log.update(lpips_error.cpu().numpy().mean(),len(output_exp))

        message = 'RMSE_test: {loss1.ave:.5f} || PSNR_test: {loss2.ave:.5f} || SSIM_test: {loss3.ave:.5f} || LPIPS_test: {loss4.ave:.5f}'.format(loss1=rmse_log,loss2=psnr_log,loss3=ssim_log,loss4=lpips_log)
        LOGGER.info(message)
    LOGGER.info('Finish Testing')




def test(checkpoint=''):
    args = get_args()
    if checkpoint:
        args.load_model = checkpoint
    LOGGER = ConsoleLogger('test_'+args.trial, 'test')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    # dataset============================================================
    test_set = PA_dataset(args.test_data_root,args.test_data_meta_root,args,stage='Test')
    test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16, drop_last=False)
    LOGGER.info('Initial Dataset Finished')
    # ====================================================================
    model = UPAMNet(inner_channel=args.inner_channel,norm_groups=args.norm_groups,channel_mults=args.channel_mults,attn_res=args.attn_res,res_blocks=args.res_blocks)
    LOGGER.info('Initial Model Finished')

    device = torch.device(f'cuda:{args.gpu}')
    model = model.cuda(device)

    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    rmse_log = AverageMeter()
    psnr_log = AverageMeter()
    ssim_log = AverageMeter()
    lpips_log = AverageMeter()
    loss_lpips = lpips.LPIPS(net='alex').cuda(device)
    with torch.no_grad():
        model.eval()
        for it, batch in enumerate(test_dataloader, 0):
            rgb = batch['input'].to(torch.float32).cuda(device)
            gt = batch['gt'].to(torch.float32).cuda(device)
            down = batch['down'].to(torch.float32).cuda(device)
            batch_size = len(rgb)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
            # =========feed into the network=========================
            output = rgb + model(rgb)
            for idx in range(len(output)):
                rmse = calc_rmse(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(),  minmax[idx])
                rmse_log.update(rmse, 1)
                rmse = calc_rmse(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                rmse_log.update(rmse, 1)
                psnr = calc_psnr(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                psnr_log.update(psnr, 1)
                ssim = calc_ssim(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy(), minmax[idx])
                ssim_log.update(ssim, 1)

            output_exp = output.expand(len(output), 3, rgb.shape[2], rgb.shape[3])
            output_exp = torch.clip(output_exp, -1, 1)
            gt_exp = gt.expand(len(gt), 3, rgb.shape[2], rgb.shape[3])
            gt_exp = torch.clip(gt_exp, -1, 1)
            lpips_error = loss_lpips(output_exp, gt_exp)
            lpips_log.update(lpips_error.cpu().numpy().mean(), len(output_exp))

        message = 'RMSE_test: {loss1.ave:.5f} || PSNR_test: {loss2.ave:.5f} || SSIM_test: {loss3.ave:.5f} || LPIPS_test: {loss4.ave:.5f}'.format(
            loss1=rmse_log, loss2=psnr_log, loss3=ssim_log, loss4=lpips_log)
        LOGGER.info(message)
        LOGGER.info('Fininshing.')




def test_single_image(checkpoint=''):

    def save_results_color(gt, pred, down, save_root_prefix, minmax=[0, 1], gray=1):
        import matplotlib.pyplot as plt
        image_pred = pred * (minmax[1] - minmax[0]) + minmax[0]
        image_pred[image_pred < 0] = 0
        # image_pred = np.log(255 * (image_pred/np.max(image_pred)))
        # image_pred[image_pred > 255] = 255
        image_gt = gt * (minmax[1] - minmax[0]) + minmax[0]
        image_gt[image_gt < 0] = 0
        # image_gt[image_gt>255]=255
        image_down = down * (minmax[1] - minmax[0]) + minmax[0]
        image_down[image_down < 0] = 0
        # image_down[image_down > 255] = 255
        if gray == 1:
            image_pred = np.array(image_pred, dtype=np.uint8)
            image_pred = Image.fromarray(image_pred)
            image_pred.save(save_root_prefix + '_pred.png')

            image_gt = np.array(image_gt, dtype=np.uint8)
            image_gt = Image.fromarray(image_gt)
            image_gt.save(save_root_prefix + '_gt.png')

            image_down = np.array(image_down, dtype=np.uint8)
            image_down = Image.fromarray(image_down)
            image_down.save(save_root_prefix + '_bi.png')

        if gray == 0:
            image_pred = np.array(image_pred, dtype=np.float64)
            plt.imsave(save_root_prefix + '_pred.png', image_pred, cmap='hot')

            image_gt = np.array(image_gt, dtype=np.float64)
            plt.imsave(save_root_prefix + '_gt.png', image_gt, cmap='hot')

            image_down = np.array(image_down, dtype=np.float64)
            plt.imsave(save_root_prefix + '_bi.png', image_down, cmap='hot')

    args = get_args()
    if checkpoint:
        args.load_model = checkpoint
    LOGGER = ConsoleLogger(args.trial, 'test')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)

    # ====================================================================
    # prepare for dataset
    img_folder = ''
    img_listdir = sorted(os.listdir(img_folder))
    img_root_list = []
    for idx in range(len(img_listdir)):
        img_root_list.append(os.path.join(img_folder,img_listdir[idx]))

    # model
    model = UPAMNet(inner_channel=args.inner_channel, norm_groups=args.norm_groups,
                             channel_mults=args.channel_mults, attn_res=args.attn_res, res_blocks=args.res_blocks)
    LOGGER.info('Initial Model Finished')
    device = torch.device(f'cuda:{args.gpu}')
    model = model.cuda(device)

    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')
    with torch.no_grad():
        model.eval()
        for idx in tqdm(range(len(img_root_list))):
            img_root = img_root_list[idx]
            img_info = np.array(Image.open(img_root))
            h,w = img_info.shape
            img_s = np.max([h,w])
            interval = 32
            scale = 1
            if img_s % interval == 0:
                img_info = np.array(Image.open(img_root).resize((img_s*scale,img_s*scale),Image.Resampling.BICUBIC))
            else :
                img_s = ((img_s // interval) + 1) * interval
                img_info = np.array(Image.open(img_root).resize((img_s*scale,img_s*scale),Image.Resampling.BICUBIC))

            d_min = np.min(img_info)
            d_max = np.max(img_info)
            img_info = (img_info - d_min)/(d_max-d_min)
            img_info = img_info.reshape(1,img_s*scale,img_s*scale)
            img_info = np.transpose(img_info,[2,0,1])
            input = torch.Tensor(img_info).to(torch.float32).cuda(device)
            input = input.view(1,1,img_s*scale,img_s*scale)
            output = input + model(input)
            save_root = os.path.join(logdir, str(idx).zfill(5))
            save_results_color(output[0,0].cpu().numpy(),output[0,0].cpu().numpy(),input[0,0].cpu().numpy(),save_root, np.array([d_min,d_max]), 0)
        LOGGER.info('Fininshing.')



if __name__ == "__main__":
    args = get_args()

    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()
    if args.mode == 'test_offline':
        test_single_image(checkpoint='')


#






