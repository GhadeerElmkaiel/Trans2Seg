import os
import sys
import torch
import numpy as np
import time

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


def demo():
    args = parse_args()
    args.test = True

    ##############################################
    # args.config_file = "configs/trans10kv2/trans2seg/trans2seg_medium_all_sber.yaml"
    # args.input_img = "tools/4.jpg"
    ##############################################

    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path

    ##############################################
    # cfg.TEST.TEST_MODEL_PATH = "./workdirs/trans10kv2/trans2seg_medium/Sber2400_50_All_classes.pth"
    # cfg.DATASET.NAME = "sber_dataset_all"
    ##############################################

    # cfg.TEST.TEST_MODEL_PATH = args.model_path
    cfg.check_and_freeze()
    print(cfg.TEST.TEST_MODEL_PATH)

    default_setup(args)

    # output folder
    output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'vis_result_{}_{}_{}_{}'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir+"/Flipped")
        os.makedirs(output_dir+"/Orig")
        os.makedirs(output_dir+"/Merged_mask")
        os.makedirs(output_dir+"/Orig_confidence")
        os.makedirs(output_dir+"/Flipped_confidence")


    confidence_flipped_dir=output_dir+"/Flipped_confidence"
    confidence_output_dir=output_dir+"/Orig_confidence"
    flipped_dir=output_dir+"/Flipped"
    merged_dir=output_dir+"/Merged_mask"
    output_dir=output_dir+"/Orig"

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    # softmax layer for confidence
    softmax_layer = torch.nn.Softmax(dim=0)

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
            
        # Get the mask
        pred = torch.argmax(output[0], 1)[0].cpu().data.numpy()
        pred_f = torch.argmax(output[0], 1)[1].cpu().data.numpy()
        output_norm = softmax_layer(output[0])
        confidence_all = torch.max(output_norm, 1)[0].cpu().data.numpy()*255
        
        mask = get_color_pallete(pred, cfg.DATASET.NAME)
        mask_f = get_color_pallete(pred_f, cfg.DATASET.NAME)
        mask = mask.resize(size_)
        mask_f = mask_f.resize(size_)
        mask_f = mask_f.transpose(method = Image.FLIP_LEFT_RIGHT)

        cropping_results = [np.array(mask), np.array(mask_f)]
        cropping_edges = [[0, 0, size_[0], size_[1]], [0, 0, size_[0], size_[1]]]

        outname = os.path.splitext(os.path.split(img_path)[-1])[0] + '.png'
        mask.save(os.path.join(output_dir, outname))
        mask_f.save(os.path.join(flipped_dir, outname))

        # Calc confidence
        confidence = np.array(confidence_all, dtype=np.uint8)[0]
        conf_img = Image.fromarray(confidence)
        conf_img = conf_img.resize(size_, Image.BILINEAR)
        conf_img.save(os.path.join(confidence_output_dir, outname))


        confidence_f = np.array(confidence_all, dtype=np.uint8)[1]
        conf_img_f = Image.fromarray(confidence_f)
        conf_img_f = conf_img_f.resize(size_, Image.BILINEAR)
        conf_img_f = conf_img_f.transpose(method = Image.FLIP_LEFT_RIGHT)
        conf_img_f.save(os.path.join(confidence_flipped_dir, outname))
        cropping_confidence = [conf_img, conf_img_f]
        # mask_f.save(os.path.join(flipped_dir, outname))

        merged_result = getMergedSemanticFromCrops(cropping_results, cropping_confidence, cropping_edges, "And", [size_[1], size_[0]])
        merged_result.save(os.path.join(merged_dir, outname))



def getMergedSemanticFromCrops(crops_result, crops_confidence, crops_edges, function, full_size):
    palette_mirror = 0
    palette_glass = 1
    palette_OOS = 3
    palette_floor = 4
    palette_FU = 2
    palette_BG = 5

    #################################################################
    ####################### Using And function ######################
    #################################################################

    if function.lower() == "and":
        # rospyLogInfoWrapper("Using And for merging cropped images")
        orig_glass = crops_result[0]==palette_glass
        orig_mirror = crops_result[0]==palette_mirror
        orig_OOS = crops_result[0]==palette_OOS
        orig_floor = crops_result[0]==palette_floor
        orig_FU = crops_result[0]==palette_FU
        # classes = [np.ones(full_size) for _ in range(max_clasess)]
        # merged_classes = []
        for i in range(1, len(crops_result)):
            cropped_glass = crops_result[i]==palette_glass
            cropped_mirror = crops_result[i]==palette_mirror
            cropped_OOS = crops_result[i]==palette_OOS
            cropped_floor = crops_result[i]==palette_floor
            cropped_FU = crops_result[i]==palette_FU
            cropped_all_optical = np.logical_or(cropped_mirror, cropped_glass)
            cropped_all_optical = np.logical_or(cropped_all_optical, cropped_OOS)
            cropped_all_floor = np.logical_or(cropped_FU, cropped_floor)

            cropped_all_optical_extended = np.ones(full_size)
            cropped_all_optical_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_all_optical

            cropped_all_floor_extended = np.ones(full_size)
            cropped_all_floor_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_all_floor

            orig_glass = np.logical_and(cropped_all_optical_extended, orig_glass)
            orig_mirror = np.logical_and(cropped_all_optical_extended, orig_mirror)
            orig_OOS = np.logical_and(cropped_all_optical_extended, orig_OOS)
            orig_floor = np.logical_and(cropped_all_floor_extended, orig_floor)
            orig_FU = np.logical_and(cropped_all_floor_extended, orig_FU)

        background = np.logical_or(orig_glass, orig_mirror)
        background = np.logical_or(background, orig_OOS)
        background = np.logical_or(background, orig_floor)
        background = np.logical_or(background, orig_FU)
        background = np.logical_not(background)
        res = orig_glass * palette_glass
        res += orig_mirror * palette_mirror
        res += orig_OOS * palette_OOS
        res += orig_floor * palette_floor
        res += orig_FU * palette_FU
        res += background * palette_BG

        res_img = get_color_pallete(res, cfg.DATASET.NAME)

        res_img = res_img.convert("RGB")
        return res_img

    #################################################################
    ####################### Using Or function #######################
    #################################################################

    elif function.lower() == "or":
        # rospyLogInfoWrapper("Using And for merging cropped images")
        orig_glass = crops_result[0]==palette_glass
        orig_mirror = crops_result[0]==palette_mirror
        orig_OOS = crops_result[0]==palette_OOS
        orig_floor = crops_result[0]==palette_floor
        orig_FU = crops_result[0]==palette_FU
        # classes = [np.ones(full_size) for _ in range(max_clasess)]
        # merged_classes = []
        for i in range(1, len(crops_result)):
            cropped_glass = crops_result[i]==palette_glass
            cropped_mirror = crops_result[i]==palette_mirror
            cropped_OOS = crops_result[i]==palette_OOS
            cropped_floor = crops_result[i]==palette_floor
            cropped_FU = crops_result[i]==palette_FU
            # cropped_all_optical = np.logical_or(cropped_mirror, cropped_glass)
            # cropped_all_optical = np.logical_or(cropped_all_optical, cropped_OOS)
            # cropped_all_floor = np.logical_or(cropped_FU, cropped_floor)

            cropped_glass_extended = np.zeros(full_size)
            cropped_glass_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_glass

            cropped_mirror_extended = np.zeros(full_size)
            cropped_mirror_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_mirror

            cropped_OOS_extended = np.zeros(full_size)
            cropped_OOS_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_OOS

            cropped_floor_extended = np.zeros(full_size)
            cropped_floor_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_floor

            cropped_FU_extended = np.ones(full_size)
            cropped_FU_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = cropped_FU

            orig_glass = np.logical_or(cropped_glass_extended, orig_glass)
            orig_free = np.logical_not(orig_glass)
            orig_mirror = np.logical_and(np.logical_or(cropped_mirror_extended, orig_mirror), orig_free)
            orig_free = np.logical_and(orig_free, np.logical_not(orig_mirror))
            orig_OOS = np.logical_and(np.logical_or(cropped_OOS_extended, orig_OOS), orig_free)
            orig_free = np.logical_and(orig_free, np.logical_not(orig_OOS))
            orig_floor = np.logical_and(np.logical_or(cropped_floor_extended, orig_floor), orig_free)
            orig_FU = np.logical_and(cropped_FU_extended, orig_FU)

        background = np.logical_or(orig_glass, orig_mirror)
        background = np.logical_or(background, orig_OOS)
        background = np.logical_or(background, orig_floor)
        background = np.logical_or(background, orig_FU)
        background = np.logical_not(background)
        res = orig_glass * palette_glass
        res += orig_mirror * palette_mirror
        res += orig_OOS * palette_OOS
        res += orig_floor * palette_floor
        res += orig_FU * palette_FU
        res += background * palette_BG

        res_img = get_color_pallete(res, cfg.DATASET.NAME)

        res_img = res_img.convert("RGB")
        return res_img

    #################################################################
    #################### Using confidence values# ###################
    #################################################################

    elif function.lower() == "confidence":

        confidence_extended_all = [crops_confidence[0]]
        results_extended_all = [crops_result[0]]
        for i in range(1, len(crops_confidence)):

            cropped_confidence_extended = np.zeros(full_size)
            cropped_confidence_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = crops_confidence[i]
            confidence_extended_all.append(cropped_confidence_extended)

            cropped_segmentation_extended = np.zeros(full_size)
            cropped_segmentation_extended[crops_edges[i][1]:crops_edges[i][3], crops_edges[i][0]:crops_edges[i][2]] = crops_result[i]
            results_extended_all.append(cropped_segmentation_extended)

        max_confidence_args = np.argsort(confidence_extended_all, axis=0)
        result_sorted_confidence = np.take_along_axis(np.array(results_extended_all), max_confidence_args, axis=0)
        res = result_sorted_confidence[-1]
        # rospyLogInfoWrapper("result_max_confidence shape"+str(result_sorted_confidence.shape))
        # rospyLogInfoWrapper("result_max_confidence[0][1]"+str(max_confidence_args[0][0][0:4]))
        # rospyLogInfoWrapper("result_max_confidence[0][2]"+str(max_confidence_args[2][0][0:4]))

        res_img = get_color_pallete(res, cfg.DATASET.NAME)

        res_img = res_img.convert("RGB")
        return res_img


if __name__ == '__main__':
    demo()
