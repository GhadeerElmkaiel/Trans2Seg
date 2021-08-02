import os
import sys
import torch
import numpy as np
import time
import re

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg


def demo(model_path, dataset_path, testset_name, epochs):
    args = parse_args()
    dataset_name = "sber_dataset_all"
    args.config_file = "configs/trans10kv2/trans2seg/trans2seg_medium_all_sber.yaml"
    args.input_img = dataset_path
    args.test = True
    # args.config_file = "configs/trans10kv2/trans2seg/trans2seg_medium.yaml"
    # args.input_img = "tools/4.jpg"
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.TEST.TEST_MODEL_PATH = model_path
    cfg.DATASET.NAME = dataset_name
    # cfg.TEST.TEST_MODEL_PATH = args.model_path
    # cfg.check_and_freeze()
    # print(cfg.TEST.TEST_MODEL_PATH)

    default_setup(args)

    # output folder
    output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'training_on_both_500/tested_on_{}/trained_for_{}'.format(
        testset_name, epochs))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir+"/Flipped")
        os.makedirs(output_dir+"/Orig")

    flipped_dir=output_dir+"/Flipped"
    output_dir=output_dir+"/Orig"

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    if os.path.isdir(args.input_img):
        img_paths = [os.path.join(args.input_img, x) for x in os.listdir(args.input_img) if "." in x]
    else:
        img_paths = [args.input_img]
    
    # print("Images for inference:")
    # print(img_paths)
    print("\n Output file path: ", output_dir)
    # print("\n CFG: /n", cfg)
    print("Dataset Name: ", cfg.DATASET.NAME)
    for img_path in img_paths:
        image = Image.open(img_path).convert('RGB')
        flipped_image = Image.open(img_path).convert('RGB')
        size_ = image.size

        flipped_image = flipped_image.transpose(method = Image.FLIP_LEFT_RIGHT)
        flipped_image = flipped_image.resize((cfg.TRAIN.BASE_SIZE, cfg.TRAIN.BASE_SIZE), Image.BILINEAR)
        flipped_image = transform(flipped_image).unsqueeze(0).to(args.device)

        image = image.resize((cfg.TRAIN.BASE_SIZE, cfg.TRAIN.BASE_SIZE), Image.BILINEAR)
        # images = transform(image).unsqueeze(0).to(args.device)
        image = transform(image).unsqueeze(0).to(args.device)

        # start_time = time.time()
        images = torch.cat((image, flipped_image), 0)
        # print("Base Size: "+str(cfg.TRAIN.BASE_SIZE))
        with torch.no_grad():
            output = model(images)
            
        # pred_time = time.time()
        # print("Model inference time: ", pred_time- start_time)
        # print(output)
        # print(len(output))

        # pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        pred = torch.argmax(output[0], 1)[0].cpu().data.numpy()
        pred_f = torch.argmax(output[0], 1)[1].cpu().data.numpy()

        # print("Argmax time: ", time.time() - pred_time)
        # print("uniques: ", np.unique(pred))
        
        # mask = get_color_pallete(pred, "sber_dataset")
        mask = get_color_pallete(pred, cfg.DATASET.NAME)
        mask_f = get_color_pallete(pred_f, cfg.DATASET.NAME)
        mask = mask.resize(size_)
        mask_f = mask_f.resize(size_)
        mask_f = mask_f.transpose(method = Image.FLIP_LEFT_RIGHT)
        outname = os.path.splitext(os.path.split(img_path)[-1])[0] + '.png'
        mask.save(os.path.join(output_dir, outname))
        mask_f.save(os.path.join(flipped_dir, outname))


if __name__ == '__main__':
    # datasets = ["Sber2400/test/images/", "Sber3500/test/images/", "SberMerged/test/images/"]
    datasets = ["Sber2400/test/images/", "Sber3500/test/images/", "SberMerged/test/images/"]
    models = [20*i for i in range(1,26)]
    for dataset in datasets:
        for model in models:
            model_name = str(model)+".pth"
            model_path = "./workdirs/trans10kv2/trans2seg_medium/Traind_on_Sber2400_and_Sber3500_for_500_epoches/" + model_name
            dataset_path = "./datasets/" + dataset
            dataset_name = re.split("[/]", dataset)[0]
            print(dataset_name)
            # dataset_name = "sber_dataset_all"
            # print("model_path: ", model_path)
            # print("dataset_path: ", dataset_path)
            demo(model_path, dataset_path, dataset_name, model)
