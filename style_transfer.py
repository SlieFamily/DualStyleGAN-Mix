import os
import numpy as np
import torch

from util import save_image, load_image

import argparse # 命令参数解析库
from argparse import Namespace

from torchvision import transforms
from torch.nn import functional as F 
import torchvision

from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp



class TestOptions():
    def __init__(self):
        # argparse 模块是 Python 内置的用于命令项选项与参数解析的模块
        # 创建一个命令行解析器对象 ——创建 ArgumentParser() 对象
        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        
        
        # 给解析器添加命令行参数 ——调用add_argument() 方法添加参数
        self.parser.add_argument("--content", type=str, default='./data/content/081680.jpg', help="原始图像(content image)的路径")
        self.parser.add_argument("--style", type=str, default='cartoon', help="目标风格名称（target style type）")
        self.parser.add_argument("--style_id", type=int, default=53, help="风格图像的ID")
        self.parser.add_argument("--truncation", type=float, default=0.75, help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="外部风格尺度")
        self.parser.add_argument("--name", type=str, default='cartoon_transfer', help="保存生成图像的文件名")
        self.parser.add_argument("--preserve_color", action="store_true", help="保持原始图像的色彩")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="模型存放地址")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="风格模型名称")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="生成图像的保存路径")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="数据集路径")
        self.parser.add_argument("--align_face", action="store_true", help="对原始图像进行人脸规格化")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="外部风格编码的名称")
        self.parser.add_argument("--wplus", action="store_true", help="使用原始 pSp encoder 进行内部风格编码的提取")


    # 定义解析参数的函数/方法
    def parse(self):

        # 自动解析命令行的参数 ——使用 parse_args() 解析添加的参数
        # 对本程序来说，相当于 self.opt 中就已经解析有成员变量：style、style_id、truncation等
        self.opt = self.parser.parse_args()

        # 设置 option 的 外部风格标识
        # 如果存在优化过的模型，就使用其文件名
        # 否则默认用 exstyle_code.npy
        if self.opt.exstyle_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'

        # vars() 函数 解析出 opt 的键值对字典
        args = vars(self.opt)

        # 依次打印所设置的参数/options
        print('加载配置中……')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        
        return self.opt



# 图像规格化
def run_alignment(args):
    import dlib # 一个机器学习的开源库
    from model.encoder.align_all_parallel import align_face #人脸规格化，作者：lzhbrian

    # 导入/下载 人脸识别68个特征点检测数据库
    modelname = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 

    predictor = dlib.shape_predictor(modelname) #返回训练好的人脸68特征点检测器

    # 通过检测点和原始图像进行规格化
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


if __name__ == "__main__":
    
    device = "cuda"

    parser = TestOptions() # 创建解析器
    args = parser.parse() # 获取参数信息

    print('*'*98)
    
    # 返回具有类型转换和归一化功能的函数transform
    transform = transforms.Compose([ # 用Compose把多个步骤整合到一起
        transforms.ToTensor(), # 把一个PIL/Numpy.ndarray类型的图片转化为tensor类型
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        # 使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
    ])
    
    # 生成器
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    generator.eval()

    # map_location=lambda storage, loc:storage 把所有的张量加载到GPU 1中
    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)
    
    # 选择编码器
    if args.wplus:
        model_path = os.path.join(args.model_path, 'encoder_wplus.pt')
    else:
        model_path = os.path.join(args.model_path, 'encoder.pt')


    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    
    # 默认输出尺寸 1024*1024
    if 'output_size' not in opts:
        opts['output_size'] = 1024    
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    # 读取外部风格数据
    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()

    z_plus_latent = not args.wplus
    return_z_plus_latent = not args.wplus
    input_is_latent = args.wplus    
    
    print('模型加载成功!')
    

    # 当前计算不需要反向传播，使用之后 with torch.no_grad()，强制后边的内容不进行计算图的构建
    with torch.no_grad(): 
        viz = []
        # 加载原始图像
        if args.align_face:
            I = transform(run_alignment(args)).unsqueeze(dim = 0).to(device)
            I = F.adaptive_avg_pool2d(I, 1024) #自适应平均池化
        else:
            I = load_image(args.content).to(device)
        viz += [I]

        # 重构原始图像及其内在风格编码
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                   z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        viz += [img_rec]

        stylename = list(exstyles.keys())[args.style_id]
        latent = torch.tensor(exstyles[stylename]).to(device)
        if args.preserve_color and not args.wplus:
            latent[:,7:18] = instyle[:,7:18]
        
        # 外部风格编码
        exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
        if args.preserve_color and args.wplus:
            exstyle[:,7:18] = instyle[:,7:18]
            
        # 加载风格图像
        S = None
        if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
            S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
            viz += [S]

        # 风格迁移
        # input_is_latent: instyle is not in W space
        # z_plus_latent: instyle is in Z+ space
        # use_res: use extrinsic style path, or the style is not transferred
        # interp_weights: weight vector for style combination of two paths
        img_gen, _ = generator([instyle], exstyle, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                              truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        viz += [img_gen]

    print('图像生成成功!')
    
    save_name = args.name+'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])

    # 保存过程图像（拼接viz中的4幅图）
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 4, 2).cpu(), 
               os.path.join(args.output_path, save_name+'_overview.jpg'))
    # 保存生成图像
    save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name+'.jpg'))

    print('图像保存成功!')
