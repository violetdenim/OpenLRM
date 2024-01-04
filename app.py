import argparse

from PIL import Image
import os.path
from rembg import remove
import gradio as gr
import numpy as np
import cv2
import torch

from lrm.inferrer import LRMInferrer



# this function crops source image to fit in 512x512 square window
def crop_512(img):
    img, alpha = img[:,:,:3], img[:,:,3]
    img.reshape(-1, 3)[alpha.reshape(-1) == 0] = [255, 255, 255]
    x, y, w, h = cv2.boundingRect(alpha)
    img = img[y:y+h, x:x+w, :]

    if max(h, w) > 512:
        scale_factor = 412.0 / max(h, w)
        new_w, new_h = int(scale_factor * w), int(scale_factor * h)
        img = cv2.resize(img, (new_w, new_h))
        h, w = img.shape[:2]

    assert(max(h, w) <= 512)
    dh, dw = (512 - h) // 2, (512 - w) // 2
    img = np.pad(img, [(dh, 512 - dh - h), (dw, 512 - dw - w), (0, 0)], constant_values=255)

    assert (img.shape[0] == 512)
    assert (img.shape[1] == 512)
    return img

# this function removes background
def get_object(src_path: str, erosion_coeff=1.5) -> np.ndarray:
    if not os.path.exists(src_path):
        print(f"No such file or directory: {src_path}")
        return None
    with Image.open(src_path) as test:
        res = remove(test, False)
        res = np.asarray(res).copy()
        if erosion_coeff > 0:
            pix_size = int(erosion_coeff * max(res.shape[0], res.shape[1]) / 224)
            res[:,:,3] = cv2.erode(res[:,:,3], np.ones((2*pix_size+1, 2*pix_size+1)))
        return res

# this function loads and preprocess image to be used in reconstruction
def load_preprocessed(path):
    test = get_object(path)
    if test is not None:
        test = crop_512(test)
    return test

def process_image(inferrer, path):
    test = load_preprocessed(path)

    def prepare_numpy(source_image: np.ndarray, source_image_size):
        image = torch.tensor(source_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        # if RGBA, blend to RGB
        if image.shape[1] == 4:
            image = image[:, :3, ...] * image[:, 3:, ...] + (1 - image[:, 3:, ...])
        image = torch.nn.functional.interpolate(image, size=(source_image_size, source_image_size), mode='bicubic',
                                                align_corners=True)
        image = torch.clamp(image, 0, 1)
        return image

    # this function repeats LRMInferrer.infer
    # but we load and unload data from\to memory, not on disk
    def demo_infer(obj: LRMInferrer, source_image: np.ndarray, source_size: int, render_size: int, mesh_size: int):
        if source_size <= 0:
            source_size = obj.infer_kwargs['source_size']
        if render_size <= 0:
            render_size = obj.infer_kwargs['render_size']

        image = prepare_numpy(source_image, source_size)
        return obj.infer_single(
            image.to(obj.device),
            render_size=render_size,
            mesh_size=mesh_size,
            export_video=True,
            export_mesh=True,
        )

    results = demo_infer(inferrer, source_image=test, source_size=-1, render_size=-1, mesh_size=384)
    mesh_path = os.path.splitext(path)[0] + '.obj'
    results["mesh"].export(mesh_path, 'obj')
    return [Image.fromarray(test), mesh_path, mesh_path]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lrm-base-obj-v1')
    parser.add_argument('--share', action='store_true', default=False)
    args = parser.parse_args()

    title = "Demo: zero-shot 3D reconstruction"
    description = "This is demo for https://github.com/violetdenim/OpenLRM (fork of https://github.com/3DTopia/OpenLRM project)"
    examples_dir = "assets/with_background"
    examples = [[os.path.join(examples_dir, file)] for file in os.listdir(examples_dir)]

    with LRMInferrer(args.model_name) as inferrer:
        demo = gr.Interface(fn=lambda x: process_image(inferrer, x),
                         inputs=[gr.Image(type="filepath", label="Input Image")],
                         outputs=[gr.Image(label="foreground", type="pil"),
                                  gr.Model3D(label="3d mesh reconstruction", clear_color=[1.0, 1.0, 1.0, 1.0]),
                                  gr.File(label="3d obj")],
                         title=title,
                         description=description,
                         examples=examples,
                         allow_flagging="never",
                         cache_examples=False)
        demo.launch(share=args.share)
