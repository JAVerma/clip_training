import clip
import wandb
import os,cv2
import numpy as np
import torch,glob
import torch.nn as nn
from torch.cuda import amp
device = "cuda" if torch.cuda.is_available() else "cpu"
path=glob.glob('/home/ai-team/members/jayant/liquify_classifier/clip-classifier/weights_type1_myntra_shoe/*')
for i in path:
    name=os.path.basename(i)[:-4]
    print(name)
    num_classes = 7
    class QuickGELU(nn.Module):
        def forward(self, x: torch.Tensor):
            return x * torch.sigmoid(1.702 * x)
    get_activation = {
        'q_gelu': QuickGELU,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
    }
    CONFIG = dict(
        clip_type='RN50',
        epochs=100,
        max_lr=3e-5,
        pct_start=0.2,
        anneal_strategy='linear',
        weight_decay=0.0002,
        batch_size=160,
        dropout=0.5,
        hid_dim=512,
        activation='relu'
    )
    model=cls_head = nn.Sequential(
        nn.Linear(512, CONFIG["hid_dim"]),
        get_activation[CONFIG["activation"]](),
        nn.Dropout(CONFIG["dropout"]),
        nn.Linear(CONFIG["hid_dim"], num_classes)
    ).to(device)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)
            self.std = 255 * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16, device=device).reshape(1, 3, 1, 1)
            self.clip_model, preprocess = clip.load(CONFIG["clip_type"], device)
            self.clip_model = self.clip_model.float()
            self.cls_head = nn.Sequential(
                # nn.Dropout(CONFIG["dropout"]),
                nn.LazyLinear(CONFIG["hid_dim"]),
                get_activation[CONFIG["activation"]](),
                nn.Dropout(CONFIG["dropout"]),
                nn.Linear(CONFIG["hid_dim"], num_classes)
            ).to(device).train()
        def forward(self, x):
            # print(x.shape)
            # x = x.unsqueeze(1)
            # x = x.repeat(1, 3, 1, 1)
            
            # print(x.shape)
            x = (x - self.mean).div_(self.std)
            x = self.clip_model.visual(x)
            x = self.cls_head(x)
            return x
    print('checked')
    model = Classifier()
    model.load_state_dict(torch.load(i))
    print('weights_loaded')
    class Wrapped_linear_model(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @torch.inference_mode()
        def forward(self, data, fp16=True):
            with amp.autocast(enabled=fp16):
                data=data.permute(0,3,1,2)
                x = self.model(data)
                x = torch.sigmoid(x)
            return x
    wrp_model = Wrapped_linear_model(model).to(device).eval()

    image=cv2.imread('/home/ai-team/members/jayant/myntra_top_classifier/footwear_data/Myntra-Footwear/Training/data/45_back/2 (52)Formal_Shoes.jpg')
    image = cv2.resize(image,(224,224))[None]
    image = torch.from_numpy(image).cuda()
    # image = image.permute(0,3,1,2)
    print(image.shape)
    # mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    # std = 255 * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
    # image = (image - mean).div_(std)
    # clip_model, preprocess = clip.load(CONFIG["clip_type"])
    # clip_model = clip_model.float()
    # image = clip_model.visual(image)
    print('checkpoint2')
    torch.cuda.synchronize()
    with torch.no_grad():
        svd_out = wrp_model(image, True)
    torch.cuda.synchronize()
    print( svd_out[0])
    with torch.inference_mode():
        traced_script_module = torch.jit.trace(wrp_model, image)
        # traced_script_module = torch.jit.optimize_for_inference(traced_script_module)
    OUT_PATH = "./out_myntra_type1_shoe"
    os.makedirs(OUT_PATH, exist_ok=True)

    traced_script_module.save(f"{OUT_PATH}/model_{name}.pt")
    traced_script_module = torch.jit.load(f"{OUT_PATH}/model_{name}.pt")
    with torch.no_grad():
        o = traced_script_module(image)
    o=o[0].cpu().numpy()
    print(o.shape, np.argmax(o))