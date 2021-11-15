import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DudeNet
from utils import *
from torchvision.utils import save_image


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DudeNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--model_path", type=str, default="abc.model_3.pth", help='path of model pth file')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--add_noise", default=False, action='store_true', help='add noise in image')

opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
	# check if cuda exist or not
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    print('Loading model ...\n')
    net = DudeNet(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join( opt.test_data, '*'))
    files_source.sort()
	
	# create images saving directory
    img_saving_path= "test_images/"
    if not os.path.exists(img_saving_path):
        os.mkdir(img_saving_path) 
		
		
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        
        if opt.add_noise:
          # noise
          torch.manual_seed(12) #set the seed
          # noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
          noise = torch.zeros(Img.shape)
          stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
          for n in range(noise.size()[0]):
            sizeN = noise[0,:,:,:].size()
            noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
          # noisy image
          # INoisy = ISource + noise
          # replacing guassian noise with speckle noise
          INoisy = ISource + (ISource * noise)
        else:
          INoisy =  torch.clone(ISource)
		  
        ISource = Variable(ISource)
        INoisy = Variable(INoisy)
        ISource= ISource.to(device) 
        INoisy = INoisy.to(device) 
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)
        
        base = os.path.basename(f)
        name = os.path.splitext(base)[0]
        save_image(ISource[0].cpu(), img_saving_path  + str(name) + '_input.png')
        if opt.add_noise:
          save_image(INoisy[0].cpu(), img_saving_path  + str(name) + '_noisy.png')
        save_image(Out[0].cpu(), img_saving_path  + str(name) + '_refined.png')
        
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
