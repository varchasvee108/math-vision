import torch
from torchvision import transforms
from PIL import Image

VOCAB = ["<SOS>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<EOS>", "<PAD>"]

stoi = {s: i for i, s in enumerate(VOCAB)}
itos = {i: s for i, s in enumerate(VOCAB)}

PAD_ID = stoi["<PAD>"]
EOS_ID = stoi["<EOS>"]
SOS_ID = stoi["<SOS>"]


class VisionMathProcessor:
    def __init__(self, image_size):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def preprocess_image(self, img):
        return self.image_transform(img)

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chars = []
        for t in token_ids:
            if t in (EOS_ID, PAD_ID):
                break
            chars.append(itos[t])
        return "".join(chars)
