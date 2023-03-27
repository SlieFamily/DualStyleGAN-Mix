import os
import numpy as np
import pathlib
import gradio as gr
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import functional as F 
import torchvision
import argparse
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp

device = "cuda"

transform = transforms.Compose([ # 用Compose把多个步骤整合到一起
    transforms.ToTensor(), # 把一个PIL/Numpy.ndarray类型的图片转化为tensor类型
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    # 使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
])

encoder = None

# 生成器
generator = DualStyleGAN(1024, 512, 8, 2, res_index = 6)
generator.eval()

# map_location=lambda storage, loc:storage 把所有的张量加载到GPU 1中
ckpt = torch.load(os.path.join('./checkpoint/', 'metfaces', 'generator-001300.pt'), map_location = lambda storage, loc: storage)
generator.load_state_dict(ckpt["g_ema"])
generator = generator.to(device)

# 外部风格编码
exstyles = np.load(os.path.join('./checkpoint/', 'metfaces', 'exstyle_code.npy'), allow_pickle='TRUE').item()



# 图像规格化
def run_alignment(img_path):

    import dlib # 一个机器学习的开源库
    from model.encoder.align_all_parallel import align_face #人脸规格化，作者：lzhbrian

    # 导入/下载 人脸识别68个特征点检测数据库
    modelname = os.path.join('./checkpoint/', 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 

    predictor = dlib.shape_predictor(modelname) #返回训练好的人脸68特征点检测器

    # 通过检测点和原始图像进行规格化
    aligned_image = align_face(filepath=img_path, predictor=predictor)
    return aligned_image


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    tensor = torch.clamp((tensor + 1) / 2 * 255, 0, 255).to(torch.uint8)
    return tensor.cpu().numpy().transpose(1, 2, 0)


# 重构原始图像及其内在风格编码
def reconstruct_face(image, encoder_type):
    # 选择编码器
    if encoder_type == 'W+ encoder':
        z_plus_latent = False
        return_z_plus_latent = False 
        model_path = os.path.join('./checkpoint/', 'encoder_wplus.pt')
    else:  
        z_plus_latent = True
        return_z_plus_latent = True 
        model_path = os.path.join('./checkpoint/', 'encoder.pt')
    
    # 加载编码器
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts['output_size'] = 1024    
    opts = argparse.Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    # 人脸重建
    image = Image.fromarray(image)
    input_data = transform(image).unsqueeze(0).to(device)

    img_rec, instyle = encoder(input_data, 
                                randomize_noise = False, 
                                return_latents = True, 
                                z_plus_latent = z_plus_latent,
                                return_z_plus_latent = return_z_plus_latent,
                                resize = False)  
    img_rec = torch.clamp(img_rec.detach(), -1, 1)
    img_rec = postprocess(img_rec[0])

    return img_rec, instyle


# 生成风格图像
def image_generate(encoder_type, style_index, structure_weight, color_weight, structure_only, instyle):
    if encoder_type == 'W+ encoder':
        z_plus_latent = False
        input_is_latent = True
    else:  
        z_plus_latent = True
        input_is_latent = False 

    stylename = list(exstyles.keys())[style_index]
    latent = torch.tensor(exstyles[stylename]).to(device)
    if structure_only and encoder_type == 'Z+ encoder':
        latent[0, 7:18] = instyle[0, 7:18]
    exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
    if structure_only and encoder_type == 'W+ encoder':
        exstyle[:,7:18] = instyle[:,7:18]

    img_gen, _ = generator([instyle], exstyle, 
                            input_is_latent = input_is_latent, 
                            z_plus_latent = z_plus_latent,
                            truncation = 0.75, 
                            truncation_latent = 0, 
                            use_res = True, 
                            interp_weights = [structure_weight]*7+[color_weight]*11)

    img_gen = torch.clamp(img_gen.detach(), -1, 1)
    img_gen = postprocess(img_gen[0])
    return img_gen

def set_example_image(example: list) -> dict:
    return gr.Image.update(value = example[0])

def show_style_image(style_index: int) -> str:
    url = list(exstyles.keys())[style_index]
    return f'./data/metfaces/{url}'



def main():
    # 指定webui的gradio主题和风格
    with gr.Blocks(theme = gr.themes.Soft(spacing_size="lg",radius_size="lg"), 
                   css='https://raw.githubusercontent.com/SlieFamily/DualStyleGAN-Mix/main/style.css'
                    ) as demo:
        
        # gr.HTML('<img src="https://s3.bmp.ovh/imgs/2023/03/23/cae0ab67b6d6b1d8.png" alt="top_image" style="margin: auto;"/>')
        
        with gr.Box():
            with gr.Row():
                encoder_type = gr.Radio(choices=['Z+ encoder', 'W+ encoder'],
                                         value='Z+ encoder',
                                          label='编码器类型')   
            
        # 预处理
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label = '输入图片',
                                               type = 'filepath')
                                             
                    with gr.Row():
                        preprocess_button = gr.Button('预处理')
                    
                with gr.Column():
                    with gr.Row():
                        aligned_face = gr.Image(label = '获取人像',
                                                type = 'numpy',
                                                interactive = False)
                with gr.Column():
                    reconstructed_face = gr.Image(label = '人脸重建',
                                                  type = 'numpy',
                                                  interactive = False)
                    instyle = gr.Variable()

            # 图像选择示例
            with gr.Row():
                paths = sorted(pathlib.Path('data/content').glob('*.jpg'))
                example_images = gr.Dataset(label = '示例图选',
                                            components = [input_image],
                                            samples = [[path.as_posix()]
                                                     for path in paths])                    

        with gr.Box():
            with gr.Row():
                with gr.Column():
                    # 显示油画图像选择预览图
                    style_index = gr.Slider(0,
                                            316,
                                            value = 26,
                                            step = 1,
                                            label = '风格图像序号')
                    style_image = gr.Image(label = '预览',
                                               type = 'filepath')                       
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        structure_weight = gr.Slider(0,
                                                     1,
                                                     value=0.6,
                                                     step=0.1,
                                                     label='结构风格权重')
                    with gr.Row():
                        color_weight = gr.Slider(0,
                                                 1,
                                                 value=1,
                                                 step=0.1,
                                                 label='色彩风格权重')
                    with gr.Row():
                        structure_only = gr.Checkbox(label='仅结构风格')
                    with gr.Row():
                        generate_button = gr.Button('生成')

                with gr.Column():
                    result = gr.Image(label='结果图',interactive=False)


        # 预处理按钮点击执行
        preprocess_button.click(fn = run_alignment,
                                inputs = [input_image],
                                outputs = aligned_face)
        # 选中样例图时
        example_images.click(fn = set_example_image,
                             inputs = example_images,
                             outputs = example_images.components)

        aligned_face.change(fn = reconstruct_face,
                            inputs = [aligned_face, encoder_type],
                            outputs = [reconstructed_face,instyle])

        # 选择风格图像
        style_index.release(fn = show_style_image,
                            inputs = [style_index.value],
                            outputs = [style_image])

        # 生成风格图像按钮点击执行
        generate_button.click(fn = image_generate,
                              inputs = [
                                  encoder_type,
                                  style_index,
                                  structure_weight,
                                  color_weight,
                                  structure_only,
                                  instyle,
                              ],
                              outputs = result)

    # 启动webui
    demo.launch(
        share = True,
    )


if __name__ == '__main__':
    main()         