import torch, gc
import cv2
import requests
import os.path
import contextlib
from PIL import Image
from modules.shared import opts, cmd_opts
from modules import processing, images, shared, devices
import os

from torchvision.transforms import Compose

from repositories.midas.midas.dpt_depth import DPTDepthModel
from repositories.midas.midas.midas_net import MidasNet
from repositories.midas.midas.midas_net_custom import MidasNet_small
from repositories.midas.midas.transforms import Resize, NormalizeImage, PrepareForNet

import numpy as np

def load_model(device, model_path, model_type="dpt_large_384", optimize=True, size=None, square=False):
    """Load the specified network.

    Args:
        device (device): the torch device used
        model_path (str): path to saved model
        model_type (str): the type of the model to be loaded
        optimize (bool): optimize the model to half-integer on CUDA?
        size (int, int): inference encoder image size
        square (bool): resize to a square resolution?

    Returns:
        The loaded network, the transform which prepares images as input to the network and the dimensions of the
        network input
    """
    if "openvino" in model_type:
        from openvino.runtime import Core

    keep_aspect_ratio = not square

    if model_type == "dpt_beit_large_512":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_512",
            non_negative=True,
        )
        net_w, net_h = 512, 512
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitb16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2l24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2b24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_tiny_256":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swinl12_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_next_vit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="next_vit_large_6m",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # We change the notation from dpt_levit_224 (MiDaS notation) to levit_384 (timm notation) here, where the 224 refers
    # to the resolution 224x224 used by LeViT and 384 is the first entry of the embed_dim, see _cfg and model_cfgs of
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/levit.py
    # (commit id: 927f031293a30afb940fff0bee34b85d9c059b0e)
    elif model_type == "dpt_levit_224":
        model = DPTDepthModel(
            path=model_path,
            backbone="levit_384",
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )
        net_w, net_h = 224, 224
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21_384":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small_256":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                               non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "openvino_midas_v21_small_256":
        ie = Core()
        uncompiled_model = ie.read_model(model=model_path)
        model = ie.compile_model(uncompiled_model, "CPU")
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    if not "openvino" in model_type:
        print("Model loaded, number of parameters = {:.0f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    else:
        print("Model loaded, optimized with OpenVINO")

    if "openvino" in model_type:
        keep_aspect_ratio = False

    if size is not None:
        net_w, net_h = size

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    if not "openvino" in model_type:
        model.eval()

    if optimize and (device == torch.device("cuda")):
        if not "openvino" in model_type:
            model = model.to(memory_format=torch.channels_last)
            model = model.half()
        else:
            print("Error: OpenVINO models are already optimized. No optimization to half-float possible.")
            exit()

    if not "openvino" in model_type:
        model.to(device)

    return model, transform

class SimpleDepthMapGenerator(object):
    def calculate_depth_maps(self,image,img_x,img_y,model_type_index,invert_depth):
        try:
            model = None
            def download_file(filename, url):
                print(f"download {filename} form {url}")
                import sys
                try:
                    with open(filename+'.tmp', "wb") as f:
                        response = requests.get(url, stream=True)
                        total_length = response.headers.get('content-length')

                        if total_length is None: # no content length header
                            f.write(response.content)
                        else:
                            dl = 0
                            total_length = int(total_length)
                            for data in response.iter_content(chunk_size=4096):
                                dl += len(data)
                                f.write(data)
                                done = int(50 * dl / total_length)
                                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                                sys.stdout.flush()
                    os.rename(filename+'.tmp', filename)
                except Exception as e:
                    os.remove(filename+'.tmp')
                    print("\n--------download fail------------\n")
                    raise e

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model path and name
            model_dir = "./models/midas"
            # create path to model if not present
            os.makedirs(model_dir, exist_ok=True)
            print("Loading midas model weights ..")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models = ["dpt_beit_large_512",
                "dpt_beit_large_384",
                "dpt_beit_base_384",
                "dpt_swin2_large_384",
                "dpt_swin2_base_384",
                "dpt_swin2_tiny_256",
                "dpt_swin_large_384",
                "dpt_next_vit_large_384",
                "dpt_levit_224",
                "dpt_large_384",
                "dpt_hybrid_384",
                "midas_v21_384",
                "midas_v21_small_256",
                # "openvino_midas_v21_small_256"
            ]
            model_path = model_dir + '/' + models[model_type_index] + '.pt'
            if not os.path.exists(model_path):
                if models.index("midas_v21_384") <= model_type_index:
                    download_file(model_path, "https://github.com/isl-org/MiDaS/releases/download/v2_1/"+ models[model_type_index] + ".pt")
                elif models.index("midas_v21_384") > model_type_index > models.index("dpt_large_384"):
                    download_file(model_path, "https://github.com/isl-org/MiDaS/releases/download/v3/"+ models[model_type_index] + ".pt")
                else:
                    download_file(model_path, "https://github.com/isl-org/MiDaS/releases/download/v3_1/"+ models[model_type_index] + ".pt")
            model, transform = load_model(device, model_path, models[model_type_index], (img_x, img_y))

            img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB) / 255.0
            img_input = transform({"image": img})["image"]
            precision_scope = torch.autocast if shared.cmd_opts.precision == "autocast" and device == torch.device("cuda") else contextlib.nullcontext
            # compute
            with torch.no_grad(), precision_scope("cuda"):
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                if device == torch.device("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)
                    if not cmd_opts.no_half:
                        sample = sample.half()
                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
            # output
            depth = prediction
            numbytes=2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2**(8*numbytes))-1

            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(depth.shape)
            # single channel, 16 bit image
            img_output = out.astype("uint16")

            # # invert depth map
            if invert_depth:
                img_output = cv2.bitwise_not(img_output)

            # three channel, 8 bits per channel image
            img_output2 = np.zeros_like(image)
            img_output2[:,:,0] = img_output / 256.0
            img_output2[:,:,1] = img_output / 256.0
            img_output2[:,:,2] = img_output / 256.0
            img = Image.fromarray(img_output2)
            return img
        except Exception:
            raise
        finally:
            del model
            gc.collect()
            devices.torch_gc()
