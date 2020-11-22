from torchvision.transforms import transforms
from RandAugment import RandAugment
import cv2
import os 
from PIL import Image

path = os.path.join(os.getcwd(), 'FMD', 'image', 'train')
transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Add RandAugment with N, M(hyperparameter)
transform_train.transforms.insert(0, RandAugment(1, 9))

for root, dirs, files in os.walk(path):
    for f in files: 
        if f.endswith('.jpg'):
            for i in range(10):
                s = os.path.join(root,f)
                img = cv2.imread(s)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = transform_train(img)
                print(os.path.join(root, '{}_aug_{}'.format(i,f)))
                img.save(os.path.join(root, '{}_aug_{}'.format(i,f)))
