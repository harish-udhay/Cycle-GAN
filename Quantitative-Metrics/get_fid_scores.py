from email.mime import base
from models import Generator

import torch
from torch.autograd import Variable
import os, shutil, json
from pytorch_fid.fid_score import calculate_fid_given_paths
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils as img_utils
from datasets import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = "./out"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
gen_weights_dir = os.path.join("out", "generator")
gen_A2B = Generator(3, 3).cuda()
gen_B2A = Generator(3, 3).cuda()
real_image_dir = os.path.join(base_dir, "real_images", "train")
fakes_dir = ""
processed_data_dir = ""


def process_data(path, batch_size = 1):
    #if os.path.exists(real_image_dir):
    #    shutil.rmtree(real_image_dir)
           
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
    Tensor = torch.cuda.FloatTensor
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
        #fake_img = gen_A2B(data)
        #fake_img = gen_A2B([data for j, data in enumerate(data_loader) if i == j][0][0].to(device))
    #fake_a = gen_B2A()
    #recovered_b = 
    #recovered_a =
        #img_utils.save_image(fake_img, os.path.join(fakes_dir, "fake_" + str(j) + ".png"))

# Calculate FID score
def get_fid_score(paths, batch_size):
    return calculate_fid_given_paths(paths, batch_size, device, dims = 2048)

if __name__ == "__main__":

    #shutil.rmtree(os.path.join(real_image_dir, "real_pizza"))
    #shutil.rmtree(os.path.join(real_image_dir, "synth_pizza"))
    process_data(real_image_dir)

    fid_scores_A2A = {}
    fid_scores_B2B = {}
    fid_scores_A2B = {}
    fid_scores_B2A = {}

    # Iteratively load saved generator models and use them to generate fakes.
    # Subsequently, compare the generated fakes to test data and compute FID score
    for _, _, files in os.walk(gen_weights_dir):
        for weight_file in files:
            model = torch.load(os.path.join(gen_weights_dir, weight_file))
            #print(model.get("netG_A2B").keys())
            gen_A2B_dict = {}
            for key in model.get("netG_A2B").keys():
                gen_A2B_dict[key[7:]] = model["netG_A2B"][key]
            gen_B2A_dict = {}
            for key in model.get("netG_B2A").keys():
                gen_B2A_dict[key[7:]] = model["netG_B2A"][key]
            #print(new_dict.keys())    
            gen_A2B.load_state_dict(gen_A2B_dict)
            gen_B2A.load_state_dict(gen_B2A_dict)
            #generator.load_state_dict(model.get("Gen_Model"), strict = False)
            num_files = 100
            print(real_image_dir)
            inference_data = DataLoader(ImageDataset(root = "./out/real_images", transforms_= [transforms.ToTensor()], unaligned = True), batch_size  = 1, shuffle = True)
            #inference_data = datasets.ImageFolder( root = os.path.join(real_image_dir, "real_pizza"), transform = transforms.ToTensor())
            #data_loader = DataLoader(inference_data, batch_size = 1)
            inference(num_fakes = num_files, data_loader = inference_data)
            batch_size = 1
            epoch = model.get("epoch")
            print("Processing Epoch: {}".format(epoch))
            # Reconstructed A2A
            paths = [os.path.join(base_dir, "reconstructed", "A"), 
                    os.path.join(real_image_dir, "A")]
            score = get_fid_score(paths, batch_size)
            fid_scores_A2A[epoch] = score
            # Reconstructed B2B
            paths = [os.path.join(base_dir, "reconstructed", "B"), 
                    os.path.join(real_image_dir, "B")]
            score = get_fid_score(paths, batch_size)
            fid_scores_B2B[epoch] = score
            # Generated A2B
            paths = [os.path.join(base_dir, "generated_fakes", "B"), 
                    os.path.join(real_image_dir, "B")]
            score = get_fid_score(paths, batch_size)
            fid_scores_A2B[epoch] = score
            # Generated B2A
            paths = [os.path.join(base_dir, "generated_fakes", "A"), 
                    os.path.join(real_image_dir, "A")]
            score = get_fid_score(paths, batch_size)
            fid_scores_B2A[epoch] = score
            #epoch = int(weight_file.split("_")[2][:-4])
            #epoch = model.get("epoch")
            #print("Processing Epoch: {}".format(epoch))
            
    
    with open(os.path.join(base_dir, "fid_A2A.json"), "w") as f:
        json.dump(fid_scores_A2A, f)
    with open(os.path.join(base_dir, "fid_B2B.json"), "w") as f:
        json.dump(fid_scores_B2B, f)
    with open(os.path.join(base_dir, "fid_A2B.json"), "w") as f:
        json.dump(fid_scores_A2B, f)
    with open(os.path.join(base_dir, "fid_B2A.json"), "w") as f:
        json.dump(fid_scores_B2A, f)

    shutil.rmtree(os.path.join(base_dir, "reconstructed"))
    shutil.rmtree(os.path.join(base_dir, "generated_fakes"))
        