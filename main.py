import os
import argparse
from config.config import *

if __name__ == "__main__":
    #args
    parser = argparse.ArgumentParser(description="Volume denoising network")
    parser.add_argument('-c', '--config', type=str, default='config/config.json',
                            help='JSON file for configuration')
    args = parser.parse_args()
    
    #config
    config = get_config(args.config)

    if config["gpu_id"] !='':
        os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu_id"] 

    import torch
    import torch.nn as nn
    import torch.optim as optim 
    import time
    from datetime import datetime

    from fvcore.nn import FlopCountAnalysis

    #
    from common import *
    from dataset.data_controller import Data_Controller

    #save path
    save_folder=config['save_folder']
    model_name=config["model_name"]
    save_name='{}_{}_{}'.format(model_name, config["dataset_name"],datetime.now().strftime('%m%d%H%M'))
    save_folder=os.path.join(save_folder, save_name)
    if os.path.exists(save_folder) ==False:
        os.makedirs(save_folder)
    save_path=save_folder+"/model.pth"

    #log
    logger=get_logger(save_folder,"logging")

    # checking that cuda is available or not # 
    USE_CUDA=torch.cuda.is_available()
    DEVICE=torch.device("cuda" if USE_CUDA else "cpu")
    logger.info("CUDA: {}".format(USE_CUDA))
    config["DEVICE"]=DEVICE

    #data controller
    data_controller = Data_Controller(config)

    # get model 
    logger.info("model contructing: %s" %(save_name))
    model=get_model(model_name)
    model=model(in_channels=1,out_channels=1,data_size=data_controller.block_size).to(DEVICE)
 
    checkFLOPs(model,data_controller,logger)

    # resume state
    resume_state = config['resume_state']
    if resume_state:
        if os.path.exists(resume_state):
            model.load_state_dict(torch.load(resume_state))
            model.to(DEVICE)
            logger.info("resume state: %s" %resume_state)

    # define loss function & optimizer #
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    #train
    if config["train"]==True:
        logger.info("Training starts")
        start_epoch=config['start_epoch'];
        N_epoch=config["N_epoch"];     # epoch

        bestResult={'rmse':100,'psnr':0,'ssim':0,'epoch':0}

        total_startTime = time.time()
        for epoch in range(start_epoch, N_epoch):  # loop over the dataset multiple times   
            startTime = time.time()            
            # train model
            train(model,data_controller,criterion,optimizer)
            
            endTime = time.time() - startTime

            # calculating loss of training set & validation set
            loss_mean_test,_,_=eval(model,data_controller,criterion,data_controller.train_generator)
            logger.info('[epoch: %d, %3d %%] training set loss: %.10f '  %(epoch + 1, (epoch + 1)/N_epoch*100 , loss_mean_test))
            
            #loss_mean_test,psnr_mean,ssim_mean=eval(model,data_controller,criterion,data_controller.validation_generator)
            loss_mean_test,psnr_mean,ssim_mean=eval(model,data_controller,criterion,data_controller.test_generator, do_metric=True)
            logger.info('[epoch: %d, %3d %%] test set time: %.3f '  %(epoch + 1, (epoch + 1)/N_epoch*100  ,endTime))
            logger.info('test set rmse: %.10f,  psnr:%.10f, ssim:%.10f'  %(loss_mean_test, psnr_mean, ssim_mean))

            if ((epoch % 10) ==0) or (epoch>90):
                save_best(bestResult, epoch, loss_mean_test, psnr_mean, ssim_mean, model, save_path,logger)

        total_endTime = time.time() - total_startTime
        logger.info('Training has been finished')
        logger.info('Total time: %.3f'  %(total_endTime))

    #test
    if config["test"]==True:    
        output_folder = save_folder if (config['output']) else None   
        loss_mean_test,psnr_mean,ssim_mean=eval(model,data_controller,criterion,data_controller.test_generator, do_metric=True, output_folder=output_folder)
        logger.info('test set rmse: %.10f,  psnr:%.10f, ssim:%.10f'  %(loss_mean_test, psnr_mean, ssim_mean))

    if config["output_file"]!=None:
        output(model,data_controller,save_folder,config["output_file"])