import torch
import math
import nodes
from torch.nn.functional import affine_grid, grid_sample
import os
import hashlib
import folder_paths
from PIL import Image, ImageOps, ImageSequence
import numpy as np
from nodes import SaveImage
import folder_paths
import random

class ImageMaskTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("FLOAT", {
                    "default": 0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "y": ("FLOAT", {
                    "default": 0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "rotate": ("FLOAT", {
                    "default": 0,
                    "min": -180,
                    "max": 180,
                    "step": 0.5
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01
                }),
                "interpolation": (["nearest", "bilinear", "bicubic"], {"default": "bilinear"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "apply_transform"
    CATEGORY = "ccsun_node"

    def apply_transform(self, image, x, y, rotate, scale, interpolation, mask=None):
        # 处理没有mask输入的情况
        if mask is None:
            # 创建与image尺寸匹配的全白遮罩（假设原始图像全部可见）
            batch_size, height, width = image.shape[0], image.shape[1], image.shape[2]
            mask = torch.ones((batch_size, height, width), dtype=image.dtype, device=image.device)
            
        # 将输入转换为4D张量 (batch, height, width, channels)
        image = image.permute(0, 3, 1, 2)  # 转为NCHW格式
        mask = mask.unsqueeze(1)  # 添加通道维度

        # 调整参数方向说明
        # x: 正值→右移 负值←左移
        # y: 正值↓下移 负值↑上移
        # rotate:      正值↻顺时针 负值↺逆时针
        # scale:       >1放大 🔍 <1缩小 🔎
        
        right_left = -x  # 右左平移量
        down_up = -y     # 下上平移量

        # 调整参数方向
        angle_rad = math.radians(-rotate)  # 正角度改为顺时针旋转
        actual_scale = 1.0 / scale         # 缩放值>1时实际放大图像

        # 创建仿射变换矩阵
        cos = math.cos(angle_rad) * actual_scale
        sin = math.sin(angle_rad) * actual_scale
        
        matrix = torch.tensor([
            [cos, -sin, right_left],
            [sin, cos,  down_up]
        ], dtype=torch.float32).unsqueeze(0).repeat(image.shape[0], 1, 1)

        # 生成采样网格
        grid = affine_grid(matrix, image.size(), align_corners=False)

        # 对图像进行变换
        transformed_image = grid_sample(image, grid, mode=interpolation, padding_mode="border", align_corners=False)
        
        # 对遮罩进行变换（使用最近邻插值）
        transformed_mask = grid_sample(mask.float(), grid, mode="nearest", padding_mode="border", align_corners=False)

        # 恢复原始维度顺序
        transformed_image = transformed_image.permute(0, 2, 3, 1)  # 转回NHWC格式
        transformed_mask = transformed_mask.squeeze(1)  # 移除通道维度

        return (transformed_image, transformed_mask)
    
class LoadImage:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True})
            }
        }
    CATEGORY = "ccsun_node"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "load_image"

    def __init__(self):
        print(f"[ccsun_node] LoadImage registered in category: {self.CATEGORY}")

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        
        # 获取原始尺寸
        width, height = img.size
        
        # 计算裁剪尺寸（8的倍数）
        crop_width = max(8, (width // 8) * 8)  # 确保最小8像素
        crop_height = max(8, (height // 8) * 8)
        
        # 计算裁剪区域（居中裁剪）
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            # 居中裁剪
            cropped_image = i.crop((left, top, right, bottom)).convert('RGB')
            
            image = np.array(cropped_image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
                # 对mask进行同样的裁剪
                cropped_mask = mask[top:bottom, left:right]
            else:
                cropped_mask = torch.zeros((crop_height, crop_width), dtype=torch.float32, device='cpu')
            
            output_images.append(image)
            output_masks.append(cropped_mask.unsqueeze(0))
        
        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        
        return (output_image, output_mask, crop_width, crop_height)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return 'Invalid image file: {}'.format(image)
        return True

class severalimages(SaveImage):
    CATEGORY = "ccsun_node"  # 添加分类标识
    OUTPUT_NODE = True  # 启用输出功能
    
    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "selected_image": ("STRING", {"default": "", "multiline": False, 
                                           "placeholder": "留空显示全部，或输入1,3,5（支持多选）"})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def process_images(self, images, selected_image, prompt=None, extra_pnginfo=None):
        # 智能处理空白输入
        if not selected_image.strip():
            selected = images
        else:
            try:
                selections = [max(0, min(int(x.strip())-1, len(images)-1)) 
                            for x in selected_image.split(",") if x.strip().isdigit()]
                selected_indices = list(dict.fromkeys(selections))[:10]
                selected = images[selected_indices] if selected_indices else images
            except:
                selected = images
        
        # 只保存全部图片预览
        preview_result = super().save_images(images, filename_prefix="preview",
                                           prompt=prompt, extra_pnginfo=extra_pnginfo)
        
        return {"ui": preview_result["ui"], "result": (selected,)}

    # 添加输出说明
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_images",)
    FUNCTION = "process_images"
    OUTPUT_NODE = True

class SingleImage(SaveImage):
    CATEGORY = "ccsun_node"  # 添加分类标识
    OUTPUT_NODE = True  # 启用输出功能
    
    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "selected_image": ("INT", {"default": 1, "min": 1, 
                                        "max": 9999, "step": 1,
                                        "dynamicMax": True})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def process_images(self, images, selected_image, prompt=None, extra_pnginfo=None):
        max_index = len(images)
        selected_index = max(0, min(selected_image - 1, max_index - 1))
        
        selected = images[selected_index:selected_index+1]
        
        preview_result = super().save_images(
            images, 
            filename_prefix=f"preview_{selected_index+1}_of_{max_index}",
            prompt=prompt, 
            extra_pnginfo=extra_pnginfo
        )
        
        return {"ui": preview_result["ui"], "result": (selected,)}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_image",)
    FUNCTION = "process_images"
    OUTPUT_NODE = True

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Image Editing": ImageMaskTransform,
    "resize to 8": LoadImage,
    "several images": severalimages,
    "Single Image": SingleImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Editing": "Image Editing",
    "resize to 8": "resize to 8",
    "several images": "several images",
    "Single Image": "Single Image"
}