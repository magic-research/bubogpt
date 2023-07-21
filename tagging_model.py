import torch
import torch.nn as nn
from torchvision.transforms import transforms

from ram.models import ram


class TaggingModule(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        import gc
        self.device = device
        image_size = 384
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # load RAM Model
        self.ram = ram(
            pretrained='checkpoints/ram_swin_large_14m.pth',
            image_size=image_size,
            vit='swin_l'
        ).eval().to(device)
        print('==> Tagging Module Loaded.')
        gc.collect()

    @torch.no_grad()
    def forward(self, original_image):
        print('==> Tagging...')
        img = self.transform(original_image).unsqueeze(0).to(self.device)
        tags, tags_chinese = self.ram.generate_tag(img)
        print('==> Tagging results: {}'.format(tags[0]))
        return [tag for tag in tags[0].split(' | ')]
