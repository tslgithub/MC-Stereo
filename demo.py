from __future__ import print_function, division
import sys
sys.path.append('core')
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.mc_stereo import MCStereo, autocast
# import stereo_datasets as datasets
from utils.utils import InputPadder
import os,glob,cv2
from core.utils import frame_utils


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_image(img1,img2):
    img1 = frame_utils.read_gen(img1)
    img2 = frame_utils.read_gen(img2)
    img1 = np.array(img1).astype(np.uint8)[..., :3]
    img2 = np.array(img2).astype(np.uint8)[..., :3]
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

    return img1, img2

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()

    output_directory = args.output_directory
    os.makedirs(output_directory,exist_ok=True)

    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        image1,image2 = load_image(imfile1,imfile2)

        left_cv_image  = cv2.imread(imfile1)
        right_cv_image = cv2.imread(imfile1)

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            if iters == 0:
                flow_pr = model(image1, image2, iters=iters, test_mode=True)
            else:
                _, flow_pr = model(image1, image2, iters=iters, test_mode=True)

            disp = flow_pr.cpu().numpy()
            disp = padder.unpad(disp)

            file_stem = imfile1.split('/')[-1]
            name,end = file_stem.split(".")
            filename = os.path.join(output_directory, name+"_depth."+end )

            # 合并显示
            disp = np.round(disp * 256).astype(np.uint16)
            depth = cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_JET)
            depth = np.append(np.append(left_cv_image,right_cv_image,1),depth,1)

            cv2.rectangle(depth,(0,0),  (160,70),(0,255,0),-1)
            cv2.rectangle(depth,(640,0),(640+200,70),(0,255,0),-1)
            cv2.rectangle(depth,(640+640,0),(640+640+210,70),(0,255,0),-1)
            cv2.putText(depth,"left",  (20, 50), 5,3, (255,0,255),3)
            cv2.putText(depth,"right", (0+640,50),5,3,(255,0,255),3)
            cv2.putText(depth,"depth", (0+640+640,50),5,3,(255,0,255),3)
            ih,iw,ic = depth.shape
            cv2.line(depth,(int(iw/3),0),(int(iw/3),ih),(255,0,128),4,1 )
            cv2.line(depth,(int(iw/3)*2,0),(int(iw/3)*2,ih),(255,0,128),4,2 )

            cv2.imwrite(filename,depth , [int(cv2.IMWRITE_PNG_COMPRESSION), 0] )

            cv2.imshow("depth",depth)
            cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt',   help="restore checkpoint",default='ckpt/mc-stereo.pth')
    parser.add_argument('--dataset',        help="dataset for evaluation", default='things',choices=["eth3d", "kitti15", "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters',    type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--feature_extractor', choices=["resnet", "convnext"], default='convnext')
    parser.add_argument('--hidden_dims',    nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")
    parser.add_argument('--n_downsample',   type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru',  action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers',   type=int, default=3, help="number of hidden GRU levels")

    parser.add_argument('-l', '--left_imgs',  help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")

    args = parser.parse_args()

    model = torch.nn.DataParallel(MCStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")
    use_mixed_precision = False

    validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
