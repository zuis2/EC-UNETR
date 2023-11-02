import torch


import logging
import numpy as np

#models
from models.DL.DL_aug import UnetGenerator_3d
from models.unet.unet import UNet
from models.unet.unet_2plus import UNet_2Plus
from models.unet.unet_3plus import UNet_3Plus
from models.unetr.unetr import UNETR
from models.unetr.swin_unetr import SwinUNETR
from models.unetr_pp.unetr_pp import UNETR_PP

from models.ec_unetr.ec_unetr_u import EC_UNETR_Unweighted
from models.ec_unetr.ec_unetr_w import EC_UNETR_Weighted
from models.ec_unetr.ec_unetr_sk import EC_UNETR_Sk
from models.ec_unetr.ec_unetr_res import EC_UNETR_Res
from models.ec_unetr.ec_unetr_dense import EC_UNETR_DENSE
from models.ec_unetr.ec_unetr_sfuc import EC_UNETR_SFUC
from models.ec_unetr.ec_unetr_sfus import EC_UNETR_SFUS

from monai import metrics
from monai.metrics.regression import SSIMMetric,PSNRMetric
from fvcore.nn import FlopCountAnalysis

 

#get model by name
def get_model(name):
    return {
        'DL':UnetGenerator_3d,
        'UNet':UNet,
        'UNet_2Plus': UNet_2Plus,
        'UNet_3Plus': UNet_3Plus,
        'UNETR': UNETR,
        'SwinUNETR': SwinUNETR,
        'UNETR_PP': UNETR_PP,

        'EC_UNETR_U':EC_UNETR_Unweighted,
        'EC_UNETR_W':EC_UNETR_Weighted,
        'EC_UNETR_Sk':EC_UNETR_Sk,
        'EC_UNETR_Res':EC_UNETR_Res,
        'EC_UNETR_DENSE':EC_UNETR_DENSE,
        'EC_UNETR_SFUC':EC_UNETR_SFUC,
        'EC_UNETR_SFUS':EC_UNETR_SFUS
    }[name]

def get_logger(log_dir, name):
    logger = logging.getLogger(name)
    file_path = "{}/{}.log".format(log_dir, name)
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def metric(outputs,targets):
        x = targets # ground truth
        y = outputs # prediction
        data_range = x.max().unsqueeze(0)

        ssim = SSIMMetric(data_range=data_range,spatial_dims=3)._compute_metric(x,y)
        psnr = PSNRMetric(max_val=data_range,reduction="mean")._compute_metric(x,y)
        return psnr,ssim

#output file
def output(model,data_controller, save_folder,file):
    model.eval()
    with torch.no_grad():
            inputs=data_controller.load_file(file)
            outputs = model(inputs)

            data_controller.output_file(save_folder, 0, outputs)

#eval model
def eval(model,data_controller,criterion,generator, do_metric=False, output_folder=None):
    model.eval()
    with torch.no_grad():
        loss_sum_test=0
        loss_all=np.array((0,0,0,0,0))
        sum_psnr=0
        sum_ssim=0
        for j, (inputs, targets,index) in enumerate(generator):
            inputs,targets=data_controller.get_data(inputs,targets)
            outputs = model(inputs)

            loss_test = criterion(outputs, targets)
            loss_sum_test += (loss_test.item())**0.5 # MSE -> RMSE

            if output_folder !=None:
                data_controller.output_file(output_folder, index.data[0],outputs)

            if do_metric:
                psnr,ssim=metric(outputs,targets)
                sum_psnr+=psnr.item()
                sum_ssim+=ssim.item()
        loss_all=loss_all/(j+1)
        #print(loss_all)
        return loss_sum_test/(j+1),sum_psnr/(j+1),sum_ssim/(j+1)

#train model
def train(model,data_controller,criterion,optimizer):
    running_loss = 0.0
    model.train()
    for i, (inputs, targets,index) in enumerate(data_controller.train_generator):
        # input & target data
        inputs,targets=data_controller.get_data(inputs,targets)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # train imformation
        running_loss += (loss.item())**0.5

def save_best(bestResult, epoch, loss_mean_test, psnr_mean, ssim_mean, model, save_path,logger):

        if bestResult['rmse']<loss_mean_test:
            return
        if bestResult['psnr']>psnr_mean:
            return
        if bestResult['ssim']>ssim_mean:
            return
        
        # save the result #
        torch.save(model.state_dict(), save_path)
        logger.info('saving model: %s' %(save_path))
        bestResult['rmse']=loss_mean_test
        bestResult['psnr']=psnr_mean
        bestResult['ssim']=ssim_mean
        bestResult['epoch']=epoch

def checkFLOPs(model,data_controller,logger):
    #params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_res = (1, data_controller.block_size, data_controller.block_size, data_controller.block_size)
    input = torch.ones(()).new_empty((1, *input_res), dtype=next(model.parameters()).dtype,
                                        device=next(model.parameters()).device)
    flops = FlopCountAnalysis(model, input)
    model_flops = flops.total()
    logger.info(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
    logger.info(f"MAdds: {round(model_flops * 1e-9, 2)} G")  


