import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import torch
from numpy import random
import numpy as np
import torchvision.transforms as transforms

from lib.config import cfg
from lib.models import get_net
# from lib.core.general import non_max_suppression, scale_coords
# from lib.utils import plot_one_box
from lib.utils import plot_one_box, show_seg_result, letterbox_for_img

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

class ObjOpt:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def run_detection(frame):
    opt = {
        'weights': 'YOLOP/weights/End-to-end.pth',
        'source': frame,
        'img_size': 320,
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        # 'device': 'cpu',
        # 'save-dir': 'inference/output',
        'augment': False,
        'update': False
    }
    opt = ObjOpt(**opt)

    # device = select_device(logger,opt.device)
    # if os.path.exists(opt.save_dir):  # output dir
    #     shutil.rmtree(opt.save_dir)  # delete dir
    # os.makedirs(opt.save_dir)  # make new dir
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    
    device = torch.device('cpu')

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    # if half:
    #     model.half()  # to FP16

    # Set Dataloader
    # if opt.source.isnumeric():
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(opt.source, img_size=opt.img_size)
    # else:
    #     dataset = LoadImages(opt.source, img_size=opt.img_size)


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # vid_path, vid_writer = None, None
    # img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    def p(img0):
        h0, w0 = img0.shape[:2]

        img, ratio, pad = letterbox_for_img(img0, new_shape=opt.img_size, auto=True)
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0, shapes
    
    img, img_det, shapes = p(frame)
    

    # for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
    img = transform(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img.float()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    det_out, da_seg_out,ll_seg_out = model(img)
    # inf_out, _ = det_out

    _, _, height, width = img.shape
    pad_w, pad_h = shapes[1][1]
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = shapes[1][0][1]

    # Apply NMS
    # det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
    # det = det_pred[0]

    da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    
    ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

    # if len(det):
    #     det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
    #     for *xyxy,conf,cls in reversed(det):
    #         label_det_pred = f'{names[int(cls)]} {conf:.2f}'
    #         plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)

    return img_det