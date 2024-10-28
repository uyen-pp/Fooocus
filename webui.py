import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json
import cv2
import numpy as np

def get_task(*args):
    args = list(args)
    args.pop(0)
    
    return worker.AsyncTask(args=args)

def paste_image_in_mask(mask_img, pasted_img):
    # Step 1: Load the images
    if pasted_img is None or mask_img is None:
        return None

    if 'image' in pasted_img:
        pasted_img = pasted_img['image']

    if 'image' in mask_img:
        mask_img = mask_img['image']
    
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    mask_img = cv2.bitwise_not(mask_img)
    
    # Step 2: Find the bounding box of the masked area
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No masked area found in the mask image.")
    
    # Assuming we take the largest contour if multiple contours are found
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Step 3: Calculate the aspect ratio and resize the pasted image
    paste_h, paste_w = pasted_img.shape[:2]
    paste_aspect = paste_w / paste_h
    mask_aspect = w / h
    
    if paste_aspect > mask_aspect:
        # Image is wider than the mask area, fit by width
        new_h = h
        new_w = int(new_h * paste_aspect)
    else:
        # Image is taller than the mask area, fit by height
        new_w = w
        new_h = int(new_w / paste_aspect)

    resized_paste = cv2.resize(pasted_img, (new_w, new_h))
    
    # Step 4: Create an empty canvas to place the image on top
    result = np.zeros_like(mask_img)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels to match pasted image
    
    # Step 5: Calculate position to center the image within the bounding box
    offset_x = x - (new_w - w) // 2
    offset_y = y - (new_h - h) // 2
    
    # Step 6: Place the resized image in the masked area
    result[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_paste
    
    # Optional Step 7: Combine the result with the original mask area
    mask_3channel = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel to combine with result
    final_output = cv2.bitwise_and(mask_3channel, result)  # Apply mask to paste
    
    return final_output

def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
    from extras.inpaint_mask import generate_mask_from_image

    extras = {}
    sam_options = None
    if mask_model == 'u2net_cloth_seg':
        extras['cloth_category'] = cloth_category
    elif mask_model == 'sam':
        sam_options = SAMOptions(
            dino_prompt=dino_prompt_text,
            dino_box_threshold=box_threshold,
            dino_text_threshold=text_threshold,
            dino_erode_or_dilate=dino_erode_or_dilate,
            dino_debug=dino_debug,
            max_detections=int(sam_max_detections),
            model_type=sam_model
        )

    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

    return mask

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                print(percentage)

            # if flag == 'results':
            #     print(flag)
               
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            print(filepath)
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return

def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images

def inpaint_mode_change(mode, inpaint_engine_version):

    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]

def refresh_seed():
    return random.randint(constants.MIN_SEED, constants.MAX_SEED)

reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

import json
import numpy as np
from PIL import Image
from glob import glob
from os import path as osp

with open('args.json', 'r') as f:
    args = json.load(f)

style = 'vintage'
prompt_dir = f'prompt_images/{style}/'
pics_dir = 'pics'

with open(f'prompt_images/{style}.json', 'r') as f:
    map_obj = json.load(f)    

def update_history_link(output_format):
    if args_manager.args.disable_image_log:
        return gr.update(value='')

    return get_current_html_path(output_format)

def get_mask(size, coord):

    size = size[::-1] + [3]
    result = np.ones(size, dtype=np.uint8) * 255
    result = cv2.fillPoly(result, np.array([coord]), 0)

    return result

def get_input_image(in_pic, coord, out_size):
    # in_pic: np.array
    # coord: target coordinates
    # out_size: expected aspect ratio

    dst = np.float32(coord)
    src = np.float32([[0, 0], [in_pic.shape[1], 0], [in_pic.shape[1], pic.shape[0]], [0, in_pic.shape[0]]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(in_pic, matrix, out_size)

    return result

for path in glob(pics_dir+'/*'):

    # The for-sale photo
    pic = np.array(Image.open(path))

    for ip, v in map_obj.items():
        
        img_prompt_path = prompt_dir+ip
        if not osp.exists(img_prompt_path):
            continue

        mask_coord = v['coord']
        aspect_ratio = v['size']

        prompt = np.array(Image.open(img_prompt_path))
        mask = get_mask(aspect_ratio, mask_coord)
        masked = cv2.bitwise_and(prompt, prompt, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
        
        inp_img = get_input_image(pic, mask_coord, aspect_ratio)
        inp_img = cv2.bitwise_or(masked, inp_img)

        Image.fromarray(mask).save('/content/'+'mask_'+ip)
        Image.fromarray(inp_img).save('/content/'+ip)

        args['inpaint_input_image'] = {'image': inp_img, 'mask': mask}
        args['inpaint_mask_image_upload'] = {'image': mask, 'mask': mask}
        args['ip_image_1'] = prompt
        args['aspect_ratios_selection'] = f'{aspect_ratio[0]}×{aspect_ratio[1]}'
        args['seed'] = refresh_seed()
        if v['describe'] == '':
            args['prompt'] = "sharp focus, detailed textures, high-definition, 8k, cinematic lighting, intricate facial features, soft shadows, studio-like realism photo of " + v['describe']
        
        ctrls=list(args.values())

        task = worker.AsyncTask(args=ctrls)
        generate_clicked(task=task)