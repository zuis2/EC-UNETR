#data controller
import torch
from torch.utils import data
from dataset.dataset import *

class Data_Controller():

    def __init__(self, config):
        super(Data_Controller, self).__init__()
        
        self.DEVICE=config["DEVICE"]
        self.block_size=config["block_size"]

        #folder of all datasets
        folder=config["datasets_path"]
        dataset_name=config["dataset_name"]
        #self.data_size=144
        INPUT_PATH=folder+dataset_name+"/Pt_input"
        TARGET_PATH=folder+dataset_name+"/Pt_target"        
        
        OUTPUT_PATH='./Pt_inputdata'
        OUTPUT_FILE_NAME='./Pt_output_1'
        OUTPUT_INSIDE_NAME='output'

        #self.block_size = self.data_size
        
        #########################################################
        ### input parameter for training ###
        N_of_data=1000;  # training set. number
        N_of_vdata=100;  # validation set, number
        N_of_tdata=100;  # test set, number
        ###

        ### data loading setting ###
        batch_size=config["batch_size"]
        params1 = {'INPUT_FILE_NAME':'Pt_input', 'TARGET_FILE_NAME' : 'Pt_target','INPUT_INSIDE_NAME':'GF_Vol', 'TARGET_INSIDE_NAME':'target'}
        params2 = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 12}
        params3 = {'batch_size': 1, 'shuffle': True, 'num_workers': 12}

        N_of_start=1
        train_ID_set = range(N_of_start,N_of_start+N_of_data);
        validation_ID_set = range(N_of_start+N_of_data,N_of_start+N_of_data+N_of_vdata)
        test_ID_set = range(N_of_start+N_of_data+N_of_vdata,N_of_start+N_of_data+N_of_vdata+N_of_tdata)

        #dataset
        train_dataset = Dataset(train_ID_set,INPUT_PATH, TARGET_PATH, image_shift(4), **params1)
        validation_dataset = Dataset(validation_ID_set,INPUT_PATH, TARGET_PATH, image_shift(4), **params1)
        test_dataset = Dataset(test_ID_set,INPUT_PATH, TARGET_PATH, None, **params1)

        #generator
        self.train_generator = data.DataLoader(train_dataset, **params2)
        self.validation_generator = data.DataLoader(validation_dataset, **params3)
        self.test_generator = data.DataLoader(test_dataset, **params3)
        ###

    def get_data(self, inputs,target):
        '''
        if self.data_size != self.block_size:
            inputs,target=self.get_random_patch(inputs,target)
        '''
        inputs= inputs.view(-1,1,self.block_size,self.block_size,self.block_size).float().to(self.DEVICE);
        target= target.view(-1,1,self.block_size,self.block_size,self.block_size).float().to(self.DEVICE);
        return inputs,target

    def get_random_patch(self, inputs,target):
        data_size= self.block_size
        start_size=[0,0,0]
        for dim in range(0,3):
            full_size = inputs.shape[dim+1]
            start_size[dim]=np.random.randint(full_size-data_size)

        inputs=inputs[:,start_size[0]:start_size[0]+data_size,start_size[1]:start_size[1]+data_size,start_size[2]:start_size[2]+data_size]
        target=target[:,start_size[0]:start_size[0]+data_size,start_size[1]:start_size[1]+data_size,start_size[2]:start_size[2]+data_size]
        return inputs,target
    
    def load_file(self, file):
        input_data = scipy.io.loadmat(file)["GF_Vol"]#["ESTvol"]
        input_data= torch.tensor(input_data).view(-1,1,self.block_size,self.block_size,self.block_size).float().to(self.DEVICE);
        return input_data

    def output_file(self, output_folder, index, outputs):
        outputs = outputs.data[0][0].cpu().numpy()
        scipy.io.savemat('{}/Pt_output_{}.mat'.format(output_folder,index), {'output':outputs}) # save