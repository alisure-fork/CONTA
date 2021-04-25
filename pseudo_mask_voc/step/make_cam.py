import os
import torch
import warnings
import numpy as np
from PIL import Image
import voc12.dataloader
import torch.nn.functional as F
from torch.backends import cudnn
from alisuretool.Tools import Tools
from misc import torchutils, imutils
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from net.resnet50_cam import CAM as resnet50_cam_net
cudnn.enabled = True
warnings.filterwarnings("ignore")


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count() * args.process_times
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id // args.process_times):
        model.cuda()
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            valid_label = torch.nonzero(pack['label'][0])[:, 0]
            size = pack['size']

            result_filename = Tools.new_dir(os.path.join(
                args.cam_out_dir, img_name + '.npy'))

            ##########################################################################################################
            if os.path.exists(result_filename):
                continue
            if len(valid_label) == 0:
                Tools.print("{}-{}/{} {} {} {}".format(
                    process_id, iter, len(data_loader), os.path.basename(result_filename), valid_label, size))
                continue
            if iter % 100 == 0:
                Tools.print("{}-{}/{} {}".format(
                    process_id, iter, len(data_loader), os.path.basename(result_filename)))
                pass
            ##########################################################################################################

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True)).detach().cpu() for img in pack['img']]

            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear',
                                                               align_corners=False)[0] for o in outputs]), 0)
            strided_cam = strided_cam[valid_label]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear',
                                                               align_corners=False) for o in outputs], 0), 0)[:, 0, :size[0], :size[1]]
            highres_cam = highres_cam[valid_label]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            if args.save_cam:
                im_cam = Image.fromarray(np.asarray(strided_cam[0] * 255, dtype=np.uint8))
                im_high_res = Image.fromarray(np.asarray(highres_cam[0] * 255, dtype=np.uint8))
                try:
                    im_cam.save(Tools.new_dir(result_filename.replace("cam", "cam_vis").replace(".npy", "_cam.bmp")))
                    im_high_res.save(Tools.new_dir(result_filename.replace("cam", "cam_vis").replace(".npy", "_high_res.bmp")))
                except Exception:
                    pass
                pass

            # save cams
            np.save(result_filename, {"keys": valid_label, "cam": strided_cam, "high_res": highres_cam.numpy()})
            pass
        pass

    pass


def run(args):
    n_gpus = torch.cuda.device_count() * args.process_times

    model = resnet50_cam_net()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(
        args.train_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    Tools.print('Start make cam')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    Tools.print('End make cam')

    torch.cuda.empty_cache()
    pass
