# app.py — CIFAR-10 image recogniser (Hugging Face Space)
# ----------------------------------------------------------------
# This is the file Hugging Face runs to build your web app.
# It loads the model YOU trained (cifar_resnet.pt) and puts an
# "upload a photo -> get a prediction" page in front of it.

import torch
import torchvision
import torchvision.transforms as T
import gradio as gr

# ----------------------------------------------------------------
# 1) The model definition.
#    This is a standard torchvision ResNet18, with the final fully
#    connected layer replaced to output 10 classes (transfer learning).
#    This MUST match the architecture used in training, otherwise the
#    saved weights won't fit.
# ----------------------------------------------------------------
model = torchvision.models.resnet18(num_classes=10)

# The 10 classes CIFAR-10 knows about — IN THIS ORDER (0..9).
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ----------------------------------------------------------------
# 2) Load the trained weights once, when the app starts up.
# ----------------------------------------------------------------
model.load_state_dict(torch.load("cifar_resnet.pt", map_location="cpu"))
model.eval()  # eval mode = "we're predicting, not training"

# ----------------------------------------------------------------
# 3) Turn ANY uploaded photo into what the model expects.
#    This ResNet18 uses the standard ImageNet-style stem (7x7 conv,
#    stride 2, followed by a maxpool), so it expects larger inputs
#    than raw 32x32 CIFAR images, plus ImageNet normalization stats
#    (since it was initialised from ImageNet-pretrained weights).
# ----------------------------------------------------------------
preprocess = T.Compose([
    T.Resize((224, 224)),                # match the stem's expected input size
    T.ToTensor(),                        # -> tensor with values 0..1, shape (3, 224, 224)
    T.Normalize(
        mean=[0.485, 0.456, 0.406],      # ImageNet mean
        std=[0.229, 0.224, 0.225],       # ImageNet std
    ),
])


def predict(image):
    if image is None:
        return {}
    image = image.convert("RGB")         # make sure we have 3 colour channels
    x = preprocess(image).unsqueeze(0)   # add a batch dimension: (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]  # turn scores into probabilities
    # Gradio's Label widget wants {class_name: probability}
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ----------------------------------------------------------------
# 4) Build the web page.
# ----------------------------------------------------------------
description = (
    "Upload an image and my ResNet18 will guess which of 10 categories it is: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.\n\n"
    "⚠️ This model was trained/fine-tuned on the CIFAR-10 dataset. It will "
    "confidently guess on ANY image — even ones it should not recognise. "
    "Try different photos and notice when it succeeds and when it fails!"
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Top guesses"),
    title="My CIFAR-10 Image Recogniser (ResNet18)",
    description=description,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
