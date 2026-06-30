# app.py — CIFAR-10 image recogniser (Hugging Face Space)
# ----------------------------------------------------------------
# This is the file Hugging Face runs to build your web app.
# It loads the model YOU trained (cifar_cnn.pt) and puts an
# "upload a photo -> get a prediction" page in front of it.

import torch
import torch.nn as nn
import torchvision.transforms as T
import gradio as gr

# ----------------------------------------------------------------
# 1) The model definition.
#    This MUST be the exact same CifarCNN from your training notebook,
#    otherwise the saved weights won't fit. (Copied from Part A.)
# ----------------------------------------------------------------
class CifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# The 10 classes CIFAR-10 knows about — IN THIS ORDER (0..9).
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ----------------------------------------------------------------
# 2) Load the trained weights once, when the app starts up.
# ----------------------------------------------------------------
model = CifarCNN()
model.load_state_dict(torch.load("cifar_cnn.pt", map_location="cpu"))
model.eval()  # eval mode = "we're predicting, not training"

# ----------------------------------------------------------------
# 3) Turn ANY uploaded photo into what the model expects:
#    a 32x32 COLOUR (RGB) image.
# ----------------------------------------------------------------
preprocess = T.Compose([
    T.Resize((32, 32)),                  # shrink/stretch to 32x32
    T.ToTensor(),                        # -> tensor with values 0..1, shape (3, 32, 32)
])


def predict(image):
    if image is None:
        return {}
    image = image.convert("RGB")         # make sure we have 3 colour channels
    x = preprocess(image).unsqueeze(0)   # add a batch dimension: (1, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]  # turn scores into probabilities
    # Gradio's Label widget wants {class_name: probability}
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ----------------------------------------------------------------
# 4) Build the web page.
# ----------------------------------------------------------------
description = (
    "Upload an image and my CNN will guess which of 10 categories it is: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.\n\n"
    "⚠️ This model only ever saw tiny 32×32 colour images during training "
    "(CIFAR-10). It will confidently guess on ANY image — even ones it should not "
    "recognise. Try different photos and notice when it succeeds and when it fails!"
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Top guesses"),
    title="My CIFAR-10 Image Recogniser",
    description=description,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
