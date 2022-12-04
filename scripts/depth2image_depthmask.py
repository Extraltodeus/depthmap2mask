from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed
from modules.shared import opts, cmd_opts, state
from PIL import Image, ImageOps
from math import ceil
import cv2

import modules.scripts as scripts
from modules import sd_samplers
from random import randint, shuffle
import random
from skimage.util import random_noise
import gradio as gr
import numpy as np
import sys
import os
import copy
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Script(scripts.Script):
    def title(self):
        return "Depth aware img2img mask"
    
    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img: return

        treshold = gr.Slider(minimum=0, maximum=255, step=1, label='Contrasts cut level', value=0)
        match_size = gr.Checkbox(label="Match input size",value=True)
        net_width = gr.Slider(minimum=64, maximum=2048, step=64, label='Net width', value=384)
        net_height = gr.Slider(minimum=64, maximum=2048, step=64, label='Net height', value=384)
        with gr.Row():
            invert_depth = gr.Checkbox(label="Invert DepthMap",value=False)
            save_depthmap = gr.Checkbox(label='Save depth map', value=False)
            save_alpha_crop = gr.Checkbox(label='Save alpha crop', value=False)
            override_mask_blur = gr.Checkbox(label='Override mask blur to 0', value=True)
            override_fill = gr.Checkbox(label='Override inpaint to original', value=True)
            clean_cut     = gr.Checkbox(label='Turn the depthmap into absolute black/white', value=False)
        model_type = gr.Dropdown(label="Model", choices=['dpt_large','midas_v21','midas_v21_small'], value='midas_v21_small', type="index", elem_id="model_type")
        # model_type = gr.Dropdown(label="Model", choices=['dpt_large','dpt_hybrid','midas_v21','midas_v21_small'], value='dpt_large', type="index", elem_id="model_type")
        return    [save_depthmap,treshold,match_size,net_width,net_height,invert_depth,model_type,override_mask_blur,override_fill,clean_cut, save_alpha_crop]

    def run(self,p,save_depthmap,treshold,match_size,net_width,net_height,invert_depth,model_type,override_mask_blur,override_fill,clean_cut, save_alpha_crop):
        def remap_range(value, minIn, MaxIn, minOut, maxOut):
            if value > MaxIn: value = MaxIn;
            if value < minIn: value = minIn;
            finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
            return finalValue;
        def create_depth_mask_from_depth_map(img,save_depthmap,p,treshold,clean_cut, save_alpha_crop):
            img = copy.deepcopy(img.convert("RGBA"))
            mask_img = copy.deepcopy(img.convert("L"))
            mask_datas = mask_img.getdata()
            datas = img.getdata()
            newData = []
            maxD = max(mask_datas)
            if clean_cut and treshold == 0:
                treshold = 128
            for i in range(len(mask_datas)):
                if clean_cut and mask_datas[i] > treshold:
                    newrgb = 255
                elif mask_datas[i] > treshold and not clean_cut:
                    newrgb = int(remap_range(mask_datas[i],treshold,255,0,255))
                else:
                    newrgb = 0
                newData.append((newrgb,newrgb,newrgb,255))
            img.putdata(newData)
            return img

        sdmg = module_from_file("depthmap_for_depth2img",'extensions/depthmap2mask/scripts/depthmap_for_depth2img.py')
        sdmg = sdmg.SimpleDepthMapGenerator() #import midas

        img_x = p.width  if match_size else net_width
        img_y = p.height if match_size else net_height

        d_m = sdmg.calculate_depth_maps(p.init_images[0],img_x,img_y,model_type,invert_depth)

        if treshold > 0 or clean_cut:
            d_m = create_depth_mask_from_depth_map(d_m,save_depthmap,p,treshold,clean_cut, save_alpha_crop)

        if save_depthmap:
            images.save_image(d_m, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, p=p)

        if save_alpha_crop:
            alpha_crop = p.init_images[0].copy()
            alpha_crop.putalpha(d_m.convert("L"))
            images.save_image(alpha_crop, p.outpath_samples, "alpha-crop", p.seed, p.prompt, opts.samples_format, p=p)

        p.image_mask = d_m
        if override_mask_blur: p.mask_blur  = 0
        if override_fill: p.inpainting_fill = 1
        proc = process_images(p)
        proc.images.append(d_m)
        if save_alpha_crop:
            proc.images.append(alpha_crop)
        return proc
