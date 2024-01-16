from glob import glob
import os

import clip
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.auto import tqdm

import logging

logging.basicConfig(filename="model_info1.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
transform=A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.ChannelShuffle(p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, keep_ratio=False, rotate_method='largest_box', always_apply=False, p=0.3)
      ] )
sweep_configuration = {
    "name": "food non food",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "method": "grid",
    "parameters": {"activation_function": {"values": ['elu', 'relu', 'leaky_relu', 'quick_gelu']},
                   "batch_size":{"values":[8,16,32]},
                   "lr":{"values":[3e-6,3e-5,2e-5,1e-5,2e-6]},
                   "hid_dim":{"values":[256,128]},
                   "dropout":{"values":[0.2,0.3,0.1]}},
    
}    

sweep_id = wandb.sweep(sweep_configuration)
class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

corrupt_images = 'corrupt_images.txt'
class ImageDataset(Dataset):
    def __init__(self, root_dir,transform=None,case='train'):
        self.im_paths = glob(os.path.join(root_dir, "*", "*"))
        self.imgs = dict()
        self.trans=transform
        self.case=case
        # for path in tqdm(self.im_paths):    
        #     self.imgs[path] = self.load_image(path)
        #     self.im_paths=list(self.imgs.keys())
        self.classes=["Food","non_food"]
        self.label_dict = {c: i for i, c in enumerate(self.classes)}

        

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        label = self.label_dict[im_path.split(os.sep)[-2]]
        img = self.load_image(im_path)
        if (self.trans is not None) and self.case=='train':
            img=transform(image=img)
            img=img['image']
        return img, label

    def load_image(self, fpath):
        img = cv2.imread(fpath)
        img = cv2.resize(img, (224, 224))
        return img



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
get_activation = {
        'q_gelu': QuickGELU,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
}
class Classifier(nn.Module):
    def __init__(self,CONFIG):
        super().__init__()
        self.mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=CONFIG['device']).reshape(1, 3, 1, 1)
        self.std = 255 * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16, device=CONFIG['device']).reshape(1, 3, 1, 1)
        self.clip_model, preprocess = clip.load(CONFIG["clip_type"], CONFIG['device'])
        self.clip_model = self.clip_model.float()
        self.cls_head = nn.Sequential(
            # nn.Dropout(CONFIG["dropout"]),
            # nn.LazyLinear(CONFIG["hid_dim"]),
            nn.Linear(1024, CONFIG['hid_dim']),
            get_activation[CONFIG["activation_function"]](),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(CONFIG['hid_dim'], CONFIG['hid_dim'] // 2),
            get_activation[CONFIG["activation_function"]](),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(CONFIG['hid_dim'] // 2, CONFIG['hid_dim'] // 4),
            get_activation[CONFIG["activation_function"]](),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(CONFIG["hid_dim"] // 4, 2)
        ).to(CONFIG['device']).train()
    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = x.repeat(1, 3, 1, 1)
        x=x.permute(0,3,1,2)
        x = (x - self.mean).div_(self.std)
        x = self.clip_model.visual(x)
        x = self.cls_head(x)
        return x
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "/home/jayant/Desktop/jayant/gitlab/clip-classifier/train"
    CONFIG={}
    run = wandb.init(project="food_nonfood" )
    # CONFIG['activation_function']=wandb.config.activation_function
    # CONFIG['batch_size']=wandb.config.batch_size
    # CONFIG['max_lr']=wandb.config.lr
    # CONFIG['hid_dim']=wandb.config.hid_dim
    # CONFIG['dropout']=wandb.config.dropout  
    CONFIG['activation_function']='q_gelu'
    CONFIG['batch_size']=64
    CONFIG['max_lr']=3e-5
    CONFIG['hid_dim']=512
    CONFIG['dropout']=0.2  
    CONFIG['pct_start']=0.2
    CONFIG['clip_type']='RN50'
    CONFIG["epochs"]=40
    CONFIG["anneal_strategy"]='linear'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    CONFIG['device']=device
    train_data = ImageDataset(DATA_DIR)
    indices = list(range(len(train_data)))
    split = int(np.floor(0.125 * len(train_data)))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=CONFIG['batch_size'],
                                              pin_memory=True, drop_last=False, num_workers=8)
    testloader = torch.utils.data.DataLoader(train_data, sampler=test_sampler, batch_size=CONFIG['batch_size'],
                                             pin_memory=True, drop_last=False, num_workers=8)
    

    model = Classifier(CONFIG)
    criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["max_lr"], weight_decay=CONFIG["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["max_lr"], weight_decay=0.0002)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=CONFIG["weight_decay"])
# scaler = amp.GradScaler()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr=CONFIG["max_lr"],
                                          steps_per_epoch=len(trainloader),
                                          epochs=CONFIG["epochs"],
                                          pct_start=CONFIG["pct_start"],
                                          anneal_strategy=CONFIG["anneal_strategy"]
                                          )
                                          
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("accuracy", summary="max")
    global_accuracy = 0
# model.clip_model.requires_grad_(False)
    for epoch in range(1, CONFIG["epochs"]+1):
        model.train()
        # if epoch==10:
        #     model.clip_model.requires_grad_(True)
        losses = AverageMeter()
        with tqdm(total=len(trainloader), desc=f"Epoch {epoch:>3}/{CONFIG['epochs']}") as pbar:
            for images, lbl in trainloader:
                images = images.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                pbar.update(1)
                with amp.autocast():
                    pred = model(images)
                    loss = criterion(pred, lbl)
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                scheduler.step()
                losses.update(loss.detach_(), images.size(0))
                # scaler.update()

            model.eval()
            test_losses = AverageMeter()
            accs = AverageMeter()
            with torch.no_grad():
                for images, lbl in testloader:
                    images = images.to(device, non_blocking=True)
                    lbl = lbl.to(device, non_blocking=True)
                    with amp.autocast():
                        pred = model(images)
                        loss = criterion(pred, lbl)
                        ps = pred.softmax(dim=1)
                        acc = (ps.argmax(dim=1) == lbl).float().mean()
                    test_losses.update(loss.detach_(), images.size(0))
                    accs.update(acc.detach_(), images.size(0))
            accuracy = accs.avg.item()
            print(losses.avg.item())
            info = {
                "train_loss": round(losses.avg.item(), 6),
                "test_loss": round(test_losses.avg.item(), 6),
                "accuracy": round(accuracy, 6),
                # "lr": scheduler.get_last_lr()[0],
            }
            pbar.set_postfix(info)
            wandb.log(info)
            save_dir = os.path.join('weights', f'{wandb.run.name}-{wandb.run.id}')
            os.makedirs(save_dir, exist_ok=True)
            if accuracy > global_accuracy:
                global_accuracy = accuracy
                print(f"Saving best model: {accuracy:.4f}")
                # torch.save(model.state_dict(), f"{wandb.run.dir}/best_weights_new.pth")
                # torch.save(model.state_dict(),"weights/best.pth")
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
                logger.info(f"best model accuracy:{global_accuracy}")
            # torch.save(model.state_dict(),"weights/latest.pth")
            torch.save(model.state_dict(), os.path.join(save_dir, 'latest.pth'))
if __name__=='__main__':
    train()
    # wandb.agent(sweep_id, function=train)