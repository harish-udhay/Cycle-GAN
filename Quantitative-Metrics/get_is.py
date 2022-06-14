from models import Generator

import numpy as np
from scipy.stats import entropy
from matplotlib.image import imread

import torch
from torch.autograd import Variable
import os, shutil, json
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, utils as img_utils
from datasets import ImageDataset

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
base_dir = "./out"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
gen_weights_dir = os.path.join("out", "generator")
gen_A2B = Generator(3, 3).to(device)
gen_B2A = Generator(3, 3).to(device)
real_image_dir = os.path.join(base_dir, "real_images", "train")
fakes_dir = ""
processed_data_dir = ""

def normalize(inp, tar):
    input_image = (inp / 127.5) - 1
    target_image = (tar / 127.5) - 1
    return input_image, target_image


def process_data():
        transforms_ = [
        transforms.Resize(int(256*1.12), Image.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
        
        #data_loader = DataLoader(ImageDataset("./data", transforms_ = transforms_, unaligned = True), batch_size  = 1, shuffle = True)
        data_loader = DataLoader(ImageDataset("./data", transforms_ = transforms_, unaligned = True, mode = 'test'), batch_size  = 1, shuffle = True)
        #data_loader = DataLoader(training_data, batch_size = batch_size, shuffle = True)

        real_pizza_dir = os.path.join(real_image_dir, "B")
        synth_pizza_dir = os.path.join(real_image_dir, "A")

        if not os.path.exists(real_pizza_dir):
            os.makedirs(real_pizza_dir)
        if not os.path.exists(synth_pizza_dir):
            os.makedirs(synth_pizza_dir)

        for i, data in enumerate(data_loader):
            img_utils.save_image(data["B"], os.path.join(real_pizza_dir, "real_" + str(i) + ".png"))
            img_utils.save_image(data["A"], os.path.join(synth_pizza_dir, "real_" + str(i) + ".png"))

# Generate fakes by loading saved generator models
def inference(num_fakes, data_loader):
    A2B_img_array = []
    B2A_img_array = []
    A2A_img_array = []
    B2B_img_array = []
    Tensor = torch.FloatTensor
    input_A = Tensor(1, 3, 256, 256)
    input_B = Tensor(1, 3, 256, 256)

    fakes_dir = os.path.join(base_dir, "generated_fakes")
    if not os.path.exists(fakes_dir):
        os.makedirs(os.path.join(fakes_dir, "A"))
        os.makedirs(os.path.join(fakes_dir, "B"))

    reconstruction_dir = os.path.join(base_dir, "reconstructed")
    if not os.path.exists(reconstruction_dir):
        os.makedirs(os.path.join(reconstruction_dir, "A"))
        os.makedirs(os.path.join(reconstruction_dir, "B"))

    for j, batch in enumerate(data_loader):
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        fake_B = gen_A2B(real_A)
        img_utils.save_image(fake_B, os.path.join(fakes_dir, "B", "fake_" + str(j) + ".png"))
        fake_A = gen_B2A(real_B)
        img_utils.save_image(fake_A, os.path.join(fakes_dir, "A", "fake_" + str(j) + ".png"))

        reconstructed_A = gen_B2A(fake_B)
        img_utils.save_image(reconstructed_A, os.path.join(reconstruction_dir, "A", "rec_" + str(j) + ".png"))
        reconstructed_B = gen_A2B(fake_A)
        img_utils.save_image(reconstructed_B, os.path.join(reconstruction_dir, "B", "rec_" + str(j) + ".png"))

        for dir, _, files in os.walk(os.path.join(fakes_dir, "B")):
            for img in files:
                image = imread(os.path.join(dir, img))
                image = np.moveaxis(image, -1, 0)
                A2B_img_array.append(image)

        for dir, _, files in os.walk(os.path.join(fakes_dir, "A")):
            for img in files:
                image = imread(os.path.join(dir, img))
                image = np.moveaxis(image, -1, 0)
                B2A_img_array.append(image)

        for dir, _, files in os.walk(os.path.join(reconstruction_dir, "B")):
            for img in files:
                image = imread(os.path.join(dir, img))
                image = np.moveaxis(image, -1, 0)
                B2B_img_array.append(image)

        for dir, _, files in os.walk(os.path.join(reconstruction_dir, "A")):
            for img in files:
                image = imread(os.path.join(dir, img))
                image = np.moveaxis(image, -1, 0)
                A2A_img_array.append(image)

    return [np.array(A2A_img_array), np.array(A2B_img_array), np.array(B2A_img_array), np.array(B2B_img_array)]


def get_inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == "__main__":
    process_data(real_image_dir)

    inception_scores_A2A = {}
    inception_scores_B2B = {}
    inception_scores_A2B = {}
    inception_scores_B2A = {}

    # Iteratively load saved generator models and use them to generate fakes.
    # Subsequently, use the generated fakes to compute Inception Scores
    for _, _, files in os.walk(gen_weights_dir):
        for weight_file in files:
            model = torch.load(os.path.join(gen_weights_dir, weight_file))
            
            gen_A2B_dict = {}
            for key in model.get("netG_A2B").keys():
                gen_A2B_dict[key[7:]] = model["netG_A2B"][key]
            gen_B2A_dict = {}
            for key in model.get("netG_B2A").keys():
                gen_B2A_dict[key[7:]] = model["netG_B2A"][key]
               
            gen_A2B.load_state_dict(gen_A2B_dict)
            gen_B2A.load_state_dict(gen_B2A_dict)
            #generator.load_state_dict(model.get("Gen_Model"), strict = False)
            num_files = 100
            print(real_image_dir)
            inference_data = DataLoader(ImageDataset(root = "./out/real_images", transforms_= [transforms.ToTensor()], unaligned = True), batch_size  = 1, shuffle = True)
             
            A2A_img_array, A2B_img_array, B2A_img_array, B2B_img_array = inference(num_fakes = num_files, data_loader = inference_data)
            batch_size = 1
            epoch = model.get("epoch")
            print("Processing Epoch: {}".format(epoch))
            # Reconstructed A2A
            score = get_inception_score(A2A_img_array)
            inception_scores_A2A[epoch] = score
            # Reconstructed B2B
            score = get_inception_score(B2B_img_array)
            inception_scores_B2B[epoch] = score
            # Generated A2B
            score = get_inception_score(A2B_img_array)
            inception_scores_A2B[epoch] = score
            # Generated B2A
            score = get_inception_score(B2A_img_array)
            inception_scores_B2A[epoch] = score
            
    
    with open(os.path.join(base_dir, "is_A2A.json"), "w") as f:
        json.dump(inception_scores_A2A, f)
    with open(os.path.join(base_dir, "is_B2B.json"), "w") as f:
        json.dump(inception_scores_B2B, f)
    with open(os.path.join(base_dir, "is_A2B.json"), "w") as f:
        json.dump(inception_scores_A2B, f)
    with open(os.path.join(base_dir, "is_B2A.json"), "w") as f:
        json.dump(inception_scores_B2A, f)

    shutil.rmtree(os.path.join(base_dir, "reconstructed"))
    shutil.rmtree(os.path.join(base_dir, "generated_fakes"))
        