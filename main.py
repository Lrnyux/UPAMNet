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
    # Place your PAM dataset here
    train_set = PA_dataset()
    train_dataloader = DataLoader(train_set,batch_size=args.train_batchsize,shuffle=True,num_workers=16,drop_last=True)
    val_set = PA_dataset()
    val_dataloader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16, drop_last=False)
    test_set = PA_dataset()
    test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16, drop_last=False)
    LOGGER.info('Initial Dataset Finished')
    #====================================================================

    # model
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
    # Place your PAM dataset here
    test_set = PA_dataset()
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






if __name__ == "__main__":
    args = get_args()

    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()


#






