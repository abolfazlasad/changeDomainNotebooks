import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    for frame in tqdm(range(start, end)):
        
        # prepare input image
        img_path =osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox)<1:
            # save original image if no bbox
            num_bbox = 0
        if not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
            # yolo_bbox = yolo_bbox[0]
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]



            #####################################################################################################
            # Create folder for this frame
            frame_name = os.path.splitext(os.path.basename(img_path))[0]
            frame_out_dir = os.path.join(output_folder, "smpl_outputs", frame_name)
            os.makedirs(frame_out_dir, exist_ok=True)

            # Save mesh vertices
            np.save(os.path.join(frame_out_dir, f"{frame_name}_mesh_vertices.npy"), mesh)

            # Save mesh as .obj (if faces available from smplx model)
            if hasattr(demoer.model.module, "smplx_layer"):
                faces = demoer.model.module.smplx_layer.faces
                obj_path = os.path.join(frame_out_dir, f"{frame_name}_mesh.obj")
                with open(obj_path, "w") as f:
                    for v in mesh:
                        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                    for face in faces + 1:  # obj is 1-indexed
                        f.write(f"f {face[0]} {face[1]} {face[2]}\n")

            # Save SMPL parameters (poses, shape, etc.)
            smpl_dict = {}
            for k in ["smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose",
                    "smplx_rhand_pose", "smplx_jaw_pose", "smplx_shape", "smplx_expr", "cam_trans"]:
                if k in out:
                    smpl_dict[k] = out[k].detach().cpu().numpy()[0]
            np.savez(os.path.join(frame_out_dir, f"{frame_name}_smpl.npz"), **smpl_dict)
            #####################################################################################################





            

            # render mesh
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
            
            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])), 
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
            # draw mesh
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])


    ##################################################################################################################
    # === After finishing the per-image loop ===
    motion_out_dir = os.path.join(output_folder, "motion_sequences")
    os.makedirs(motion_out_dir, exist_ok=True)

    motion_seq = {
        "root_pose": [],
        "body_pose": [],
        "lhand_pose": [],
        "rhand_pose": [],
        "jaw_pose": [],
        "shape": [],
    }

    smpl_root = os.path.join(output_folder, "smpl_outputs")
    all_frames = sorted(os.listdir(smpl_root))

    for frame_folder in all_frames:
        npz_files = [f for f in os.listdir(os.path.join(smpl_root, frame_folder)) if f.endswith("_smpl.npz")]
        if not npz_files:
            continue
        smpl_path = os.path.join(smpl_root, frame_folder, npz_files[0])
        data = np.load(smpl_path)

        def safe_get(key, default_shape):
            return data[key] if key in data else np.zeros(default_shape, dtype=np.float32)

        motion_seq["root_pose"].append(safe_get("smplx_root_pose", (3,)))
        motion_seq["body_pose"].append(safe_get("smplx_body_pose", (21, 3)))
        motion_seq["lhand_pose"].append(safe_get("smplx_lhand_pose", (15, 3)))
        motion_seq["rhand_pose"].append(safe_get("smplx_rhand_pose", (15, 3)))
        motion_seq["jaw_pose"].append(safe_get("smplx_jaw_pose", (3,)))
        motion_seq["shape"].append(safe_get("smplx_shape", (10,)))

    # Stack into arrays [T, ...]
    for k in motion_seq:
        motion_seq[k] = np.stack(motion_seq[k], axis=0)

    # Save motion clip
    seq_path = osp.join(root_dir, 'demo', "motion_sequence.npz")
    np.savez(seq_path, **motion_seq)

    print(f"✅ Saved motion sequence -> {seq_path}")
    ##################################################################################################################



if __name__ == "__main__":
    main()
