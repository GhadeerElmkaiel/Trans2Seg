from utils.metrics import *
import numpy as np
import re
import glob
import PIL.Image as Image
import pandas as pd

# only_glass = ['928.png', '913.png', '978.png', '916.png', '973.png', '997.png', '915.png', '980.png', '923.png', '957.png', '959.png', '982.png', '995.png', '965.png', '936.png', '92.png', '924.png', '937.png', '979.png', '951.png', '931.png', '948.png', '927.png', '947.png', '992.png', '993.png', '969.png', '945.png', '956.png', '954.png', '932.png', '968.png', '981.png', '974.png', '918.png', '961.png', '919.png', '91.png', '946.png', '929.png', '960.png', '939.png', '950.png', '940.png', '917.png', '994.png', '998.png', '942.png', '912.png', '922.png', '985.png', '989.png', '935.png', '987.png', '970.png', '914.png', '938.png', '984.png', '944.png', '999.png', '934.png', '97.png', '966.png', '943.png', '958.png']


columns = ["Epochs", "Glass", "Mirror", "Other optical surface", "All Optical", "Glass And Mirrors", "Floor", "FU", "All Floor"]
res_df = pd.DataFrame(columns=columns)

testing_dataset = "Sber2400"

split = "test/"
# root = "/home/ghadeer/Projects/Datasets/Sber2400/"
root = "datasets/Sber2400/"
root_to_masks = root + split
path_to_target_glass = root_to_masks+"Glass/"
path_to_target_all = root_to_masks+"All_Optical/"
path_to_target_G_M = root_to_masks+"Glass_and_Mirrors/"
path_to_target_Floor = root_to_masks+"Floor/"
path_to_target_All_Floor = root_to_masks+"All_Floor/"
path_to_target_mirror = root_to_masks+"Mirror/"
path_to_target_OOS = root_to_masks+"Other optical surface/"
path_to_target_FU = root_to_masks+"Floor under obstacle/"

mean_IoU_res = {"trans_glass": [], "trans_Mirror":[], "trans_OOS":[], "trans_all":[], "trans_G_M":[], "trans_Floor":[], "trans_FU":[], "trans_All_Floor":[]}
# mean_acc_res = {"trans_glass": [], "trans_all":[], "trans_G_M":[], "trans_Floor":[], "trans_FU":[], "trans_Mirror":[], "trans_OOS":[], "trans_All_Floor":[]}

# path_to_target_res = "/home/ghadeer/Projects/Trans2Seg/runs/visual/with_adaptive_palette-removed/"
for i in range(1,26):
    idx = i*20
    print("Calculating results for model {}.pth".format(idx))
        
    res_root = "runs/visual/training_on_both_500/tested_on_{}/trained_for_{}/".format(testing_dataset, str(idx))
    path_to_target_res = res_root+"Orig/"
    path_to_target_flipped = res_root+ "Flipped/"
    # path_to_target_and = res_root+ "And/"
    # path_to_target_or = res_root+ "Or/"

    path_to_testing = path_to_target_res

    images = []
    for img in glob.glob(path_to_testing+"*"):
        images.append(re.split('[/]',img)[-1])

    ###########################
    # images = only_glass

    IoU_res = {"trans_glass": [], "trans_all":[], "trans_G_M":[], "trans_Floor":[], "trans_FU":[], "trans_Mirror":[], "trans_OOS":[], "trans_All_Floor":[]}
    acc_res = {"trans_glass": [], "trans_all":[], "trans_G_M":[], "trans_Floor":[], "trans_FU":[], "trans_Mirror":[], "trans_OOS":[], "trans_All_Floor":[]}


    for name in images:
        all_mask = Image.open(path_to_target_all+name)
        all_mask = all_mask.convert("L")
        all_arr = np.array(all_mask, dtype="?")

        glass_mask = Image.open(path_to_target_glass+name)
        glass_mask = glass_mask.convert("L")
        glass_arr = np.array(glass_mask, dtype="?")

        G_M_mask = Image.open(path_to_target_G_M+name)
        G_M_mask = G_M_mask.convert("L")
        G_M_arr = np.array(G_M_mask, dtype="?")

        Floor_mask = Image.open(path_to_target_Floor+name)
        Floor_mask = Floor_mask.convert("L")
        Floor_arr = np.array(Floor_mask, dtype="?")

        Mirror_mask = Image.open(path_to_target_mirror+name)
        Mirror_mask = Mirror_mask.convert("L")
        Mirror_arr = np.array(Mirror_mask, dtype="?")

        OOS_mask = Image.open(path_to_target_OOS+name)
        OOS_mask = OOS_mask.convert("L")
        OOS_arr = np.array(OOS_mask, dtype="?")

        FU_mask = Image.open(path_to_target_FU+name)
        FU_mask = FU_mask.convert("L")
        FU_arr = np.array(FU_mask, dtype="?")

        All_Floor_mask = Image.open(path_to_target_All_Floor+name)
        All_Floor_mask = All_Floor_mask.convert("L")
        All_Floor_arr = np.array(All_Floor_mask, dtype="?")


        trans_mask = Image.open(path_to_testing+name)
        trans_mask = trans_mask.resize(all_mask.size)
        # trans_mask.show()
        trans_arr = np.array(trans_mask)
        img_size = trans_arr.size
        vals = np.unique(trans_arr)

        # test = Image.open(path_to_testing+"1.png")
        # test.show()
        # vals = np.unique(test)
        
        trans_glass = trans_arr==1
        trans_mirror = trans_arr==0
        trans_oos = trans_arr==3
        trans_floor = trans_arr==4
        trans_fu = trans_arr==2
        trans_optical = np.logical_or( np.logical_or(trans_glass, trans_mirror),trans_oos)
        trans_g_and_m = np.logical_or(trans_glass, trans_mirror)
        trans_all_floor = np.logical_or(trans_floor, trans_fu)
        # trans_optical = trans_arr==0
        # trans_floor = trans_arr==1
        ##############################################
        # Only Glass
        intersection_glass = np.sum(np.logical_and(glass_arr, trans_glass))
        union_glass = np.sum(np.logical_or(glass_arr, trans_glass))
        acc_all = np.sum(np.logical_not(np.logical_xor(glass_arr, trans_glass)))
        if union_glass == 0:
            iou_glass = 1
        else:
            iou_glass = intersection_glass/union_glass
            
        acc_val = acc_all/img_size
        IoU_res["trans_glass"].append(iou_glass)
        acc_res["trans_glass"].append(acc_val)

        ##############################################
        # All Optical Surfaces
        intersection_all = np.sum(np.logical_and(all_arr, trans_optical))
        union_all = np.sum(np.logical_or(all_arr, trans_optical))
        acc_all = np.sum(np.logical_not(np.logical_xor(all_arr, trans_optical)))
        if union_all == 0:
            iou_all = 1
        else:
            iou_all = intersection_all/union_all
        acc_val = acc_all/img_size
        IoU_res["trans_all"].append(iou_all)
        acc_res["trans_all"].append(acc_val)

        ##############################################
        # Glass and Mirror
        intersection_G_M = np.sum(np.logical_and(G_M_arr, trans_g_and_m))
        union_G_M = np.sum(np.logical_or(G_M_arr, trans_g_and_m))
        acc_G_M = np.sum(np.logical_not(np.logical_xor(G_M_arr, trans_g_and_m)))
        if union_G_M == 0:
            iou_G_M = 1
        else:
            iou_G_M = intersection_G_M/union_G_M
        acc_val = acc_G_M/img_size
        IoU_res["trans_G_M"].append(iou_G_M)
        acc_res["trans_G_M"].append(acc_val)

        ##############################################
        # Mirror
        intersection_Mirror = np.sum(np.logical_and(Mirror_arr, trans_mirror))
        union_Mirror = np.sum(np.logical_or(Mirror_arr, trans_mirror))
        acc_Mirror = np.sum(np.logical_not(np.logical_xor(Mirror_arr, trans_mirror)))
        if union_Mirror == 0:
            iou_Mirror = 1
        else:
            iou_Mirror = intersection_Mirror/union_Mirror
        acc_val = acc_Mirror/img_size
        IoU_res["trans_Mirror"].append(iou_Mirror)
        acc_res["trans_Mirror"].append(acc_val)

        ##############################################
        # OOS
        intersection_OOS = np.sum(np.logical_and(OOS_arr, trans_oos))
        union_OOS = np.sum(np.logical_or(OOS_arr, trans_oos))
        acc_OOS = np.sum(np.logical_not(np.logical_xor(OOS_arr, trans_oos)))
        if union_OOS == 0:
            iou_OOS = 1
        else:
            iou_OOS = intersection_OOS/union_OOS
        acc_val = acc_OOS/img_size
        IoU_res["trans_OOS"].append(iou_OOS)
        acc_res["trans_OOS"].append(acc_val)

        ##############################################
        # Floor
        intersection_Floor = np.sum(np.logical_and(Floor_arr, trans_floor))
        union_Floor = np.sum(np.logical_or(Floor_arr, trans_floor))
        acc_Floor = np.sum(np.logical_not(np.logical_xor(Floor_arr, trans_floor)))
        if union_Floor == 0:
            iou_Floor = 1
        else:
            iou_Floor = intersection_Floor/union_Floor
        acc_val = acc_Floor/img_size
        IoU_res["trans_Floor"].append(iou_Floor)
        acc_res["trans_Floor"].append(acc_val)

        ##############################################
        # Floor Under Obstacles FU
        intersection_FU = np.sum(np.logical_and(FU_arr, trans_fu))
        union_FU = np.sum(np.logical_or(FU_arr, trans_fu))
        acc_FU = np.sum(np.logical_not(np.logical_xor(FU_arr, trans_fu)))
        if union_FU == 0:
            iou_FU = 1
        else:
            iou_FU = intersection_FU/union_FU
        acc_val = acc_FU/img_size
        IoU_res["trans_FU"].append(iou_FU)
        acc_res["trans_FU"].append(acc_val)

        ##############################################
        # All Floor
        intersection_All_Floor = np.sum(np.logical_and(All_Floor_arr, trans_all_floor))
        union_All_Floor = np.sum(np.logical_or(All_Floor_arr, trans_all_floor))
        acc_All_Floor = np.sum(np.logical_not(np.logical_xor(All_Floor_arr, trans_all_floor)))
        if union_All_Floor == 0:
            iou_All_Floor = 1
        else:
            iou_All_Floor = intersection_All_Floor/union_All_Floor
        acc_val = acc_All_Floor/img_size
        IoU_res["trans_All_Floor"].append(iou_All_Floor)
        acc_res["trans_All_Floor"].append(acc_val)

        ##############################################

    for key in IoU_res.keys():
        mean_IoU_res[key].append(np.mean(IoU_res[key]))
    # res_pd = pd.DataFrame({"Epochs": idx, "Glass": np.mean(IoU_res["trans_glass"]), "Mirror": np.mean(IoU_res["trans_Mirror"]), 
    #         "Other optical surface": np.mean(IoU_res["trans_OOS"]), "All Optical": np.mean(IoU_res["trans_all"]),
    #         "Glass And Mirrors": np.mean(IoU_res["trans_G_M"]), "Floor": np.mean(IoU_res["trans_Floor"]),
    #         "FU": np.mean(IoU_res["trans_FU"]), "All Floor": np.mean(IoU_res["trans_All_Floor"])}, index=[i])
    # res_df.append(res_pd, ignore_index=True)


    # print("iou_glass: ", np.mean(IoU_res["trans_glass"]))
    # print("iou_all_optical: ", np.mean(IoU_res["trans_all"]))
    # print("iou_G_M: ", np.mean(IoU_res["trans_G_M"]))
    # print("iou_mirror: ", np.mean(IoU_res["trans_Mirror"]))
    # print("iou_oos: ", np.mean(IoU_res["trans_OOS"]))
    # print("iou_Floor: ", np.mean(IoU_res["trans_Floor"]))
    # print("iou_FU: ", np.mean(IoU_res["trans_FU"]))
    # print("iou_All_Floor: ", np.mean(IoU_res["trans_All_Floor"]))

    # print("acc_glass: ", np.mean(acc_res["trans_glass"]))
    # print("acc_all_optical: ", np.mean(acc_res["trans_all"]))
    # print("acc_G_M: ", np.mean(acc_res["trans_G_M"]))
    # # print("acc_Floor: ", np.mean(acc_res["trans_Floor"]))
    # print("acc_mirror: ", np.mean(acc_res["trans_Mirror"]))
    # print("acc_oos: ", np.mean(acc_res["trans_OOS"]))
    # print("acc_Floor: ", np.mean(acc_res["trans_Floor"]))
    # print("acc_FU: ", np.mean(acc_res["trans_FU"]))
    # print("acc_All_Floor: ", np.mean(acc_res["trans_All_Floor"]))
res_df = pd.DataFrame(mean_IoU_res)
res_df.to_csv(testing_dataset+".csv", index=False)
print("Done")
