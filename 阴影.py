import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm

seed = 42
rng = np.random.RandomState(seed)
rotate_degree = [-30, 30]
rotate_prob = 1.0

perspective_scale = 0.15
perspective_prob =  1.0


shadow_intensity = [0.3, 0.6]
shadow_num = 4
shadow_prob = 1.0


def _apply_shadow(image: np.ndarray, rng, shadow_intensity):

    h, w = image.shape[:2]
    shadow_mask = np.zeros((h, w), dtype=np.float32)


    num_shadows = rng.randint(1, shadow_num + 1)

    for _ in range(num_shadows):
        
        angle = rng.uniform(0, np.pi * 2)
        length = rng.randint(h // 4, h // 2)
        width = rng.randint(w // 12, w // 8)

        
        margin = max(length, width) // 2
        center_x = rng.randint(margin, w - margin)
        center_y = rng.randint(margin, h - margin)

        
        layers = rng.randint(2, 4)
        for layer in range(layers):
            
            scale = 1.0 - (layer * 0.2)
            curr_length = int(length * scale)
            curr_width = int(width * scale)
            shadow_alpha = rng.uniform(shadow_intensity[0], shadow_intensity[1]) * (1 - layer * 0.3)

            
            temp_mask = np.zeros((h, w), dtype=np.float32)

            
            num_vertices = rng.randint(5, 8)  
            vertices = []
            for i in range(num_vertices):
                
                theta = 2 * np.pi * i / num_vertices + rng.uniform(-0.2, 0.2)
                r1 = curr_length / 2 * (1 + rng.uniform(-0.2, 0.2))
                r2 = curr_width / 2 * (1 + rng.uniform(-0.2, 0.2))

               
                x = center_x + r1 * np.cos(theta) * np.cos(angle) - r2 * np.sin(theta) * np.sin(angle)
                y = center_y + r1 * np.cos(theta) * np.sin(angle) + r2 * np.sin(theta) * np.cos(angle)
                vertices.append([x, y])

           
            vertices = np.array(vertices, dtype=np.int32)

            
            cv2.fillConvexPoly(temp_mask, vertices, 1.0)

            
            shadow_mask = np.maximum(shadow_mask, temp_mask * shadow_alpha)

    
    blur_size = rng.choice([15, 21, 31])
    shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)

   
    noise = rng.normal(0, 0.01, shadow_mask.shape)
    shadow_mask += noise
    shadow_mask = np.clip(shadow_mask, 0, 1)

    
    if image.ndim == 3:
        shadow_mask = shadow_mask[..., np.newaxis]

    shadowed_img = image * (1 - shadow_mask)

    return shadowed_img.astype(np.uint8)



img_list = os.listdir("test_img")

if not os.path.exists("shadow/" + "shadow_img") and not os.path.exists("shadow/" + "shadow_msk"):
        os.mkdir("shadow/" + "shadow_img")
        os.mkdir("shadow/" + "shadow_msk")

        img_path = "shadow/" + "shadow_img/"
        mask_path = "shadow/" + "shadow_msk/"

img_path = "shadow/" + "shadow_img/"
mask_path = "shadow/" + "shadow_msk/"

for nm in img_list:
        #name = nm.replace("tif", "png")
        images = cv2.imread("test_img/" + nm, -1)
        #masks = cv2.imread("perspective_mask/" + nm, 0)
        img_shadow = _apply_shadow(images, rng, shadow_intensity)
        cv2.imwrite(img_path + nm.replace("tif", "png"), img_shadow)
        #cv2.imwrite(mask_path + name, mask_rot)
        
