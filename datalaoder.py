import torch
from torch.utils.data import DataLoader, Dataset
import glob,os,cv2
import albumentations as A
class dataloader(Dataset):
    def __init__(self,root):
        self.images_path=glob.glob(os.path.join(root,'train_A','*'))
        self.target_path=glob.glob(os.path.join(root,'train_B','*'))
        self.transformation=A.compose(
            [
            A.Resize(768,768,p=1, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.GridDistortion(distort_limit=0.5,num_steps=5, p=0.2,border_mode=cv2.BORDER_CONSTANT,value=(0,0,0)),
            A.ElasticTransform(alpha=5, sigma=10, alpha_affine=10, approximate=True, same_dxdy=True, p=0.2,border_mode=cv2.BORDER_CONSTANT,value=(0,0,0)),
            A.Affine( rotate=(-5, 5), shear=(-2, 2), interpolation=cv2.INTER_LINEAR, p=0.4, mode=cv2.BORDER_REFLECT),
            ],additional_targets ={'image':'image0'}
            
        )
        
        # self.back_transform = A.Compose([
                #     A.Resize(768,768,p=1, always_apply=True),
                #     A.HorizontalFlip(p=0.5),
                #     # A.GridDistortion(distort_limit=0.5,num_steps=5, p=0.2,border_mode=cv2.BORDER_CONSTANT,value=(0,0,0)),
                #     # A.ElasticTransform(alpha=5, sigma=10, alpha_affine=10, approximate=True, same_dxdy=True, p=0.2,border_mode=cv2.BORDER_CONSTANT,value=(0,0,0)),
                #     # A.Affine( rotate=(-5, 5), shear=(-2, 2), interpolation=cv2.INTER_LINEAR, p=0.4, mode=cv2.BORDER_REFLECT),
                # ], additional_targets={'image0': 'image'})
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self,idx):
        img_path=self.images_path[idx]
        tar_path=self.target_path{idx}
        img=cv2.imread(img_path)
        tar=cv2.imread(tar_path)
        transformed_img=self.transformation(image=img,image0=tar)
        img,tar=transformed_img['image'],transformed_img['image0']
        return img,tar
    