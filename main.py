import os
import sys

gpuid = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

ALGORITHM = "Efficient Net"

IMAGE_FOLDER = "/public/ss/wyx/dataset/CVSOriginImage"
SAVE_FOLDER = "./results"
LABEL_PATH = "/public/ss/wyx/dataset/3_points_system/cvs_score_for_images_v1.json"
RULE = "cvsiii"

if ALGORITHM == "TSM":
    sys.path.insert(1, "./sequence_classification/TSM")
    from sequence_classification.TSM.train_custom import run_tsm_train

    run_tsm_train(IMAGE_FOLDER, LABEL_PATH, RULE, SAVE_FOLDER)
elif ALGORITHM == "Efficient Net":
    sys.path.insert(1, "./image_classification/efficient_net")
    sys.path.insert(2, "./image_segmentation/deeplab_v3plus")
    from image_classification.efficient_net.train_custom import run_efficient_net_train

    run_efficient_net_train(IMAGE_FOLDER, LABEL_PATH, RULE, SAVE_FOLDER)
else:
    print(f"{ALGORITHM} is not supported!")
