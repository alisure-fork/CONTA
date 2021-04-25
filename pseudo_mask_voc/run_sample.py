import os
import argparse
from misc import pyutils
from alisuretool.Tools import Tools


"""
train
eval_cam_pass
{'iou': array([0.79366776, 0.42647029, 0.29414451, 0.43481525, 0.35925153, 
       0.47427129, 0.62097323, 0.54035264, 0.47403768, 0.29000049,
       0.56819908, 0.39018552, 0.4724652 , 0.49807592, 0.60358108,
       0.52667555, 0.43747871, 0.61739674, 0.44190276, 0.51170632,
       0.45300079]), 'miou': 0.48707868257244097}
eval_ins_seg_pass
0.5iou: {'ap': array([0.27933129, 0.00103266, 0.5636691 , 0.30920865, 0.1530427 ,
       0.57904853, 0.39744859, 0.55137043, 0.04581884, 0.49919254,
       0.18329573, 0.53679976, 0.55728675, 0.47981325, 0.20840433,
       0.08084337, 0.33579813, 0.32901444, 0.49749726, 0.47986656]), 'map': 0.35338914546356937}
eval_sem_seg_pass
{'iou': array([0.88073269, 0.66759029, 0.36189445, 0.78292794, 0.60520391,
       0.6282414 , 0.79558968, 0.70426591, 0.72976084, 0.33667847,
       0.79286362, 0.3994215 , 0.73692968, 0.78152904, 0.74842453,
       0.68994714, 0.54090617, 0.82388969, 0.57362865, 0.6532389 ,
       0.58957824]), 'miou': 0.6582496539955474}
"""


"""
val
eval_cam_pass
{'iou': array([0.79095478, 0.44974042, 0.25811446, 0.44935083, 0.37828806,
       0.44271365, 0.61430836, 0.53900897, 0.46178571, 0.28119338,
       0.50923677, 0.37053352, 0.46925991, 0.46408499, 0.57568523,
       0.52911526, 0.4323498 , 0.559023  , 0.44286567, 0.51492133,
       0.51806795]), 'miou': 0.4786000972254945}
eval_ins_seg_pass
0.5iou: {'ap': array([4.33539690e-01, 1.12892301e-04, 6.06009949e-01, 2.10395444e-01,
       1.97200505e-01, 4.56853979e-01, 2.72182691e-01, 5.44548704e-01,
       5.71739444e-02, 5.17991092e-01, 7.59019703e-02, 5.35665623e-01,
       4.31762938e-01, 5.92741139e-01, 2.04481780e-01, 8.34569799e-02,
       3.36350207e-01, 4.05151930e-01, 4.26759158e-01, 3.60212024e-01]), 'map': 0.3374246319977131}
eval_sem_seg_pass
{'iou': array([0.87374619, 0.6702049 , 0.29269518, 0.69082349, 0.56988052,
       0.64511787, 0.78650181, 0.71386888, 0.67892777, 0.30514737,
       0.74535526, 0.36443157, 0.72451024, 0.70302492, 0.74181094,
       0.68146596, 0.48128112, 0.75593283, 0.56171424, 0.65451146,
       0.61099822]), 'miou': 0.6310452729378939}
"""


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

    parser = argparse.ArgumentParser()

    # Dataset
    # parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    # parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str, help="voc12/train_aug.txt")
    # parser.add_argument("--chainer_eval_set", default="train", type=str)

    # parser.add_argument("--train_list", default="voc12/train.txt", type=str)
    # parser.add_argument("--infer_list", default="voc12/train.txt", type=str, help="voc12/train_aug.txt")
    # parser.add_argument("--chainer_eval_set", default="train", type=str)

    parser.add_argument("--train_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str, help="voc12/train_aug.txt")
    parser.add_argument("--chainer_eval_set", default="val", type=str)

    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0), help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=16, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8, help="the number of random walk iterations")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Step
    parser.add_argument("--save_cam", default=True)
    parser.add_argument("--train_cam_pass", default=False)
    parser.add_argument("--train_irn_pass", default=False)

    parser.add_argument("--make_cam_pass", default=False)
    parser.add_argument("--cam_to_ir_label_pass", default=False)
    parser.add_argument("--make_ins_seg_pass", default=False)
    parser.add_argument("--make_sem_seg_pass", default=True)

    parser.add_argument("--eval_cam_pass", default=False)
    parser.add_argument("--eval_ins_seg_pass", default=False)
    parser.add_argument("--eval_sem_seg_pass", default=True)

    args = parser.parse_args()

    args.voc12_root = "/media/ubuntu/4T/ALISURE/Data/SS/voc/VOCdevkit/VOC2012"
    if not os.path.exists(args.voc12_root):
        args.voc12_root = "/mnt/4T/Data/data/SS/voc/VOCdevkit/VOC2012"
        pass

    args.process_times = 1
    # args.process_times = 2

    # Path
    runner_name = "2"
    args.log_name_dir = Tools.new_dir("sess/{}".format(runner_name))
    args.cam_weights_name = "sess/{}/res50_cam.pth".format(runner_name)
    args.irn_weights_name = "sess/{}/res50_irn.pth".format(runner_name)
    args.cam_out_dir = Tools.new_dir("result/{}/cam/{}".format(
        runner_name, os.path.splitext(os.path.basename(args.train_list))[0]))
    args.ir_label_out_dir = Tools.new_dir("result/{}/ir_label/{}".format(
        runner_name, os.path.splitext(os.path.basename(args.train_list))[0]))
    args.sem_seg_out_dir = Tools.new_dir("result/{}/sem_seg/{}".format(
        runner_name, os.path.splitext(os.path.basename(args.infer_list))[0]))
    args.ins_seg_out_dir = Tools.new_dir("result/{}/ins_seg/{}".format(
        runner_name, os.path.splitext(os.path.basename(args.infer_list))[0]))

    pyutils.Logger(os.path.join(args.log_name_dir, 'sample_train_eval.log'))
    Tools.print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)
        pass

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)
        pass

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)
        pass

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)
        pass

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)
        pass

    if args.make_ins_seg_pass is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)
        pass

    if args.eval_ins_seg_pass is True:
        import step.eval_ins_seg

        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)
        pass

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)
        pass

    if args.eval_sem_seg_pass is True:
            import step.eval_sem_seg

            timer = pyutils.Timer('step.eval_sem_seg:')
            step.eval_sem_seg.run(args)
            pass

    pass

