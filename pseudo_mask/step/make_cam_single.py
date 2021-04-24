import os
import torch
import warnings
import numpy as np
from tqdm import tqdm
import voc12.dataloader
import torch.nn.functional as F
from torch.backends import cudnn
from alisuretool.Tools import Tools
from misc import torchutils, imutils
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from tools.read_info import read_image_info
from net.resnet50_cam import CAM as resnet50_cam_net
cudnn.enabled = True
warnings.filterwarnings("ignore")


def _work(model, dataset, args):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        for iter, pack in tqdm(enumerate(data_loader), total=len(data_loader)):
            img_name = pack['name'][0]
            # valid_label = torch.nonzero(pack['label'][0])[:, 0]
            valid_label = torch.where(pack['label'][0])[0]
            size = pack['size']

            if len(torch.where(pack['label'][0])[0]) == 0:
                continue

            now_name = img_name.split("Data/DET/")[1]
            result_filename = Tools.new_dir(os.path.join(args.cam_out_dir, now_name)).replace(".JPEG", ".npy")
            if os.path.exists(result_filename):
                continue
            Tools.print("{} {} {} {}".format(iter, now_name, valid_label, size))

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = []
            for img in pack['img']:
                img = img[0].cuda()
                output_one = model(img)
                outputs.append(output_one.detach().cpu())
                pass

            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear',
                                                               align_corners=False)[0] for o in outputs]), 0)
            strided_cam = strided_cam[valid_label]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear',
                                                               align_corners=False) for o in outputs], 0), 0)[:, 0, :size[0], :size[1]]
            highres_cam = highres_cam[valid_label]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(result_filename, {"keys": valid_label, "cam": strided_cam,
                                      "high_res": highres_cam.numpy()})
            pass
        pass

    pass


def run(args):
    num_classes = 200
    model = resnet50_cam_net(num_classes=num_classes).cuda()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    image_info_list = read_image_info(args.voc12_root)

    dataset = voc12.dataloader.MyVOC12ClassificationDatasetMSF(
        image_info_list, num_classes=num_classes, scales=args.cam_scales)

    Tools.print('Start make cam')
    _work(model, dataset, args)
    Tools.print('End make cam')

    torch.cuda.empty_cache()
    pass
