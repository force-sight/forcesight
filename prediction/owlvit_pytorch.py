import requests
from PIL import Image, ImageDraw, ImageFont
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

# everything should be on gpu

processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("table1.jpg") #Image.open(requests.get(url, stream=True).raw)
# texts = [["a photo of a cat", "a photo of a dog"]]
texts = [["marker", "pencil", "mouse", "keyboard", "earphones", "sunglasses", "xbox controller"]]
inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

# Retrieve predictions for the first image for the corresponding text queries
i = 0  
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

score_threshold = 0.3
#font = ImageFont.truetype("arial.ttf", 12)
draw = ImageDraw.Draw(image)

print("for loop")
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    if score >= score_threshold:
        draw.rectangle(box, outline="red", width=4)
        label_text = f"{text[label]} ({round(score.item(), 3)})"
        draw.text([box[0], box[1]-15], label_text, fill='black')
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

image.show()
