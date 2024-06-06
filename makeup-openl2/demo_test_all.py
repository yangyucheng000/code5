import os
os.system("pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.1.1/MindSpore/unified/x86_64/mindspore-2.1.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install mindcv -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install fvcore -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install fire -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install face_recognition -i https://pypi.tuna.tsinghua.edu.cn/simple")

import argparse
from pathlib import Path
from posix import ST_WRITE

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np

import faceutils as futils
from psgan import PostProcess
from setup import setup_config, setup_argparser

import cv2
import sys
import mindspore as ms


def main(save_path='./results/full_general'):
    parser = setup_argparser()
    parser.add_argument(
        "--source_path", 
        # default="/home/wangzichuan/makeup-tmm-huawei/dataset/Makeup-Complex/paper-show-pictures/final_align/non-makeup",
        default="datasets/source_test",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_dir",
        # default="/home/wangzichuan/makeup-tmm-huawei/dataset/Makeup-Complex/paper-show-pictures/final_align/makeup",
        default="datasets/reference_test",
        help="path to reference images")
    parser.add_argument(
        "--speed",
        action="store_true",
        help="test speed")
    parser.add_argument(
        "--device",
        default="cuda",
        help="device used for inference")
    parser.add_argument(
        "--model_path",
        default=
        'checkpoints/32_3073_G.ckpt',
        help="model for loading")
    
    args = parser.parse_args()
    config = setup_config(args)
    ms.set_context(device_target='GPU', device_id=0)
    config.defrost()
    config.MODEL.NET = "net_dran"
    config.TRAINING.WINDOWS = -1
    config.freeze()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = Path(save_path)

    # Using the second cpu
    inference = Inference(
        config, args.device, args.model_path)
    postprocess = PostProcess(config)

    source_paths=sorted(list(Path(args.source_path).glob("*")))
    reference_paths =sorted(list(Path(args.reference_dir).glob("*")))
    num=0
    total_r= len(reference_paths)
    total_s= len(source_paths)
    for i in range(total_s):
        source_path = source_paths[i]
        # print(i)
        print(source_path)
        if os.path.basename(source_path)=='.DS_Store':
            continue
        source = Image.open(source_path).convert("RGB")
        #np.random.shuffle(reference_paths)
        for j in range(total_r):
            reference_path = reference_paths[j]
            print(reference_path)
            if not reference_path.is_file():
                print(reference_path, "is not a valid file.")
                continue
            if os.path.basename(reference_path) == '.DS_Store':
                continue
            reference = Image.open(reference_path).convert("RGB")
            # Transfer the psgan from reference to source.
            image, face,imgs,lms_src,lms_ref, mid_results= inference.transfer(source, reference, with_face=True)
            if face == None:
                continue
            source_crop = image
            image = postprocess(source_crop, image)
            source=source.resize((256, 256), Image.ANTIALIAS)
            reference =reference.resize((256, 256), Image.ANTIALIAS)
            source_crop =source_crop.resize((256, 256), Image.ANTIALIAS)
            image =image.resize((256, 256), Image.ANTIALIAS)
            to_image = Image.new('RGB', (256*4 , 256 ))  # 创建一个新图
            
            to_image.paste(source, (0 * 256, 0* 256))
            to_image.paste(reference, (1 * 256, 0 * 256))
            to_image.paste(image, (2 * 256, 0* 256))
            to_image.paste(imgs[0], (3 * 256, 0* 256))

            save_src_name=str(os.path.basename(source_path)).split('.png')[0]
            save_ref_name=str(os.path.basename(reference_path)).split('.png')[0]
            
            save_name = save_src_name + '_' + save_ref_name + '.png'
            save_path_dst=save_path.joinpath(save_name)
            to_image.save(save_path_dst)
            num += 1
            print(num)
if __name__ == '__main__':
    main()
