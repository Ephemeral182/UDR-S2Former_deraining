import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image,make_grid
import torchvision.transforms as tfs
from PIL import Image
from easydict import EasyDict
import yaml
from tqdm import tqdm
from UDR_S2Former import *

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
     device = torch.device("cuda:0")

def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


# load yaml config
config = load_config('config/demo.yaml')

# load model
model = Transformer((config.img_size_h,config.img_size_w)).to(device)
model = load_model_weights(model, config.model_weights)

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.imgs_dir=os.listdir(os.path.join(image_paths))
        self.imgs=[os.path.join(image_paths,img) for img in self.imgs_dir]
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        image = Image.open(image_path).convert('RGB')
        name = image_path.split('/')[-1].split(".")[0]
        image = tfs.ToTensor()(image)
        return image,name

image_paths = config.image_path
output_paths = config.output_image_path

if not os.path.exists(image_paths):
   os.makedirs(image_paths)
   
if not os.path.exists(output_paths):     
   os.makedirs(output_paths)


dataset = ImageDataset(image_paths)


batch_size = 1 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

loop = tqdm(enumerate(dataloader),total=len(dataloader))
for idx,(image,name) in loop:
    with torch.no_grad():
        image = image.to(device)
        b, c, h, w = image.shape
        print('image_size:', h, w)

        # image deraining
        tile = min(config.tile, h, w)
        print('tile',tile)
        tile_overlap = config.tile_overlap
        sf = config.scale

        # stride = tile - tile_overlap
        # h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        # w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        # E1 = torch.zeros(b, c, h*sf, w*sf).type_as(haze)
        # W1 = torch.zeros_like(E1)

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E1 = torch.zeros(b, c, h*sf, w*sf).type_as(image)
        W1 = torch.zeros_like(E1)
        E2 = torch.zeros(b, c, h*sf, w*sf).type_as(image)
        W2 = torch.zeros_like(E2)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = image[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch1,_ = model(in_patch)
                out_patch1 = out_patch1[0]
                out_patch_mask1 = torch.ones_like(out_patch1)
                E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)
        output = E1.div_(W1)
        output1 = torch.cat([image,output],3)
        # save image
        save_image(output1,os.path.join(output_paths,'cat_%s.png'%(name)),normalize=False)
        save_image(output,os.path.join(output_paths,'%s.png'%(name)),normalize=False)