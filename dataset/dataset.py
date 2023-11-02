from torch.utils import data
import scipy.io
import numpy as np

### define image trainslation ###
class image_shift(object):

    def __init__(self, translation_range=0):
        #assert isinstance(translation_range, (float, tuple))
        self.translation_range = translation_range

    def __call__(self, input_data, target_data):
        inputs, target = input_data, target_data
        
        dx, dy, dz = np.random.randint(self.translation_range*2+1, size=3)-self.translation_range
        
        inputs = np.roll(inputs, dz, axis=0)
        inputs = np.roll(inputs, dy, axis=1)
        inputs = np.roll(inputs, dx, axis=2)
        
        target = np.roll(target, dz, axis=0)
        target = np.roll(target, dy, axis=1)
        target = np.roll(target, dx, axis=2)
        
        if dz>0:
            inputs[:,:,:dz] = 0
            target[:,:,:dz] = 0
        elif dz<0:
            inputs[:,:,dz:] = 0
            target[:,:,dz:] = 0
        if dy>0:
            inputs[:,:dy,:] = 0
            target[:,:dy,:] = 0
        elif dy<0:
            inputs[:,dy:,:] = 0
            target[:,dy:,:] = 0
        if dx>0:
            inputs[:dx,:, :] = 0
            target[:dx,:, :] = 0
        elif dx<0:
            inputs[dx:, :,:] = 0
            target[dx:, :,:] = 0
        return inputs, target
###


### For data loading ###
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, INPUT_PATH,TARGET_PATH,Transform=None,INPUT_FILE_NAME='Pt_input',TARGET_FILE_NAME='Pt_target',INPUT_INSIDE_NAME='GF_Vol',TARGET_INSIDE_NAME='target'):
        'Initialization'
        self.list_IDs = list_IDs
        self.input_path = INPUT_PATH
        self.target_path = TARGET_PATH
        self.transform = Transform
        self.input_file_name = INPUT_FILE_NAME
        self.target_file_name = TARGET_FILE_NAME
        self.input_inside_name = INPUT_INSIDE_NAME
        self.target_inside_name = TARGET_INSIDE_NAME

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        # ID_set
        ID = self.list_IDs[index]
        'Generates one sample of data'
        # Load data and get target_data for matlab file
        input_data = scipy.io.loadmat('{}/{}_{}.mat'.format(self.input_path,self.input_file_name,ID))[self.input_inside_name];
        target_data = scipy.io.loadmat('{}/{}_{}.mat'.format(self.target_path,self.target_file_name,ID))[self.target_inside_name];

        if self.transform:
            input_data, target_data = self.transform(input_data, target_data)

        return input_data, target_data,ID
