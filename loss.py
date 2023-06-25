import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torchvision.models as models



class mse(_Loss):
    def __init__(self, args, **kwargs):
        super(mse, self).__init__()
        self.args = args

    def forward(self, pred, gt, **kwargs):
        return {'mse_loss':F.mse_loss(pred,gt)}


class percept(_Loss):
    def __init__(self,args, **kwargs):
        super(percept,self).__init__()
        self.args = args
        self.vgg19 = models.vgg19(pretrained=True)
        # self.feature_net = torch.nn.Sequential(*list(self.vgg19.children())[0][:15])
        loss_network = nn.Sequential(*list(self.vgg19.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.feature_net = loss_network
    def forward(self,pred,gt,**kwargs):
        B,C,H,W = pred.shape
        mse_term = F.mse_loss(pred,gt)

        pred_exp = pred.expand(B,3,H,W)
        gt_exp = gt.expand(B,3,H,W)
        pred_feats = self.feature_net(pred_exp)
        gt_feats = self.feature_net(gt_exp)
        per_term = F.mse_loss(pred_feats,gt_feats)
        return {'mse_loss':mse_term,'per_loss':per_term}


class percept_patch(_Loss):
    def __init__(self,args, **kwargs):
        super(percept_patch,self).__init__()
        self.args = args
        self.vgg19 = models.vgg19(pretrained=True)
        # self.feature_net = torch.nn.Sequential(*list(self.vgg19.children())[0][:15])
        loss_network = nn.Sequential(*list(self.vgg19.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.feature_net = loss_network

    def get_patches(self,feats,kernel_size=4,step_size=2):
        B,C,H,W = feats.shape
        h_num = (H - kernel_size) // step_size + 1
        w_num = (W - kernel_size) // step_size + 1
        patch = torch.stack([feats[:,:,idx*step_size:idx*step_size+kernel_size,jdx*step_size:jdx*step_size+kernel_size] for idx in range(h_num) for jdx in range(w_num)],dim=1)
        return patch.view(B,h_num*w_num,C,-1)


    def forward(self,pred,gt,**kwargs):
        B,C,H,W = pred.shape
        mse_term = F.mse_loss(pred,gt)

        pred_exp = pred.expand(B,3,H,W)
        gt_exp = gt.expand(B,3,H,W)



        pred_feats = self.feature_net(pred_exp)
        gt_feats = self.feature_net(gt_exp)

        pred_feats_patch = self.get_patches(pred_feats)
        gt_feats_patch = self.get_patches(gt_feats)

        # per_term = F.mse_loss(pred_feats_patch,gt_feats_patch)
        per_term = F.cosine_similarity(pred_feats_patch,gt_feats_patch,dim=-1)
        per_term = 1.0 - per_term.mean()
        return {'mse_loss':mse_term,'per_loss':per_term}

class percept_patch_prior(_Loss):
    def __init__(self,args, **kwargs):
        super(percept_patch_prior,self).__init__()
        self.args = args
        self.vgg19 = models.vgg19(pretrained=True)
        # self.feature_net = torch.nn.Sequential(*list(self.vgg19.children())[0][:15])

        # for fast computation
        loss_network_object = nn.Sequential(*list(self.vgg19.features)[:31]).eval()
        loss_network_back = nn.Sequential(*list(self.vgg19.features)[:20]).eval()
        loss_network_edge = nn.Sequential(*list(self.vgg19.features)[:11]).eval()

        for param in loss_network_object.parameters():
            param.requires_grad = False
        for param in loss_network_back.parameters():
            param.requires_grad = False
        for param in loss_network_edge.parameters():
            param.requires_grad = False

        self.feature_net_object = loss_network_object
        self.feature_net_back = loss_network_back
        self.feature_net_edge = loss_network_edge

    def get_patches(self,feats,kernel_size=4,step_size=2):
        B,C,H,W = feats.shape
        h_num = (H - kernel_size) // step_size + 1
        w_num = (W - kernel_size) // step_size + 1
        patch = torch.stack([feats[:,:,idx*step_size:idx*step_size+kernel_size,jdx*step_size:jdx*step_size+kernel_size] for idx in range(h_num) for jdx in range(w_num)],dim=1)
        return patch.view(B,h_num*w_num,C,-1)


    def forward(self,pred,gt,mask=None,**kwargs):
        B,C,H,W = pred.shape
        mse_term = F.mse_loss(pred,gt)

        pred_exp = pred.expand(B,3,H,W)
        gt_exp = gt.expand(B,3,H,W)
        mask_exp = mask.expand(B, 3, H, W)

        # for mask: 0 -> background || 1 -> foreground || 2 -> edge detection
        # t0 = time.time()
        mask_exp_back = torch.zeros_like(mask_exp).to(pred.device)
        mask_exp_back[mask_exp == 0] = 1

        mask_exp_object = torch.zeros_like(mask_exp).to(pred.device)
        mask_exp_object[mask_exp == 1] = 1

        mask_exp_edge = torch.zeros_like(mask_exp).to(pred.device)
        mask_exp_edge[mask_exp == 2] = 1



        pred_feats_back = self.feature_net_back(pred_exp * mask_exp_back)
        gt_feats_back = self.feature_net_back(gt_exp * mask_exp_back)
        pred_feats_patch_back = self.get_patches(pred_feats_back)
        gt_feats_patch_back = self.get_patches(gt_feats_back)
        per_term_back = F.cosine_similarity(pred_feats_patch_back,gt_feats_patch_back,dim=-1)
        per_term_back = 1.0 - per_term_back.mean()


        pred_feats_object = self.feature_net_object(pred_exp * mask_exp_object)
        gt_feats_object = self.feature_net_object(gt_exp * mask_exp_object)
        pred_feats_patch_object = self.get_patches(pred_feats_object)
        gt_feats_patch_object = self.get_patches(gt_feats_object)
        per_term_object = F.cosine_similarity(pred_feats_patch_object, gt_feats_patch_object, dim=-1)
        per_term_object = 1.0 - per_term_object.mean()


        pred_feats_edge = self.feature_net_edge(pred_exp * mask_exp_edge)
        gt_feats_edge = self.feature_net_edge(gt_exp * mask_exp_edge)
        pred_feats_patch_edge = self.get_patches(pred_feats_edge)
        gt_feats_patch_edge = self.get_patches(gt_feats_edge)
        per_term_edge = F.cosine_similarity(pred_feats_patch_edge, gt_feats_patch_edge, dim=-1)
        per_term_edge = 1.0 - per_term_edge.mean()



        return {'mse_loss':mse_term,'per_back_loss':per_term_back,'per_object_loss':per_term_object,'per_edge_loss':per_term_edge}








