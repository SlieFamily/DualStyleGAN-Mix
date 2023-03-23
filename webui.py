import gradio as gr

def main():
    # 指定webui的gradio主题和风格
    with gr.Blocks(theme = gr.themes.Soft(spacing_size="lg",radius_size="lg"), 
                   css='style.css'
                    ) as demo:
        
        gr.HTML('<img src="https://s3.bmp.ovh/imgs/2023/03/23/cae0ab67b6d6b1d8.png" alt="top_image" style="margin: auto;"/>')
        
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
                        input_image = gr.Image(label='输入图片',
                                               type='filepath')
                    with gr.Row():
                        preprocess_button = gr.Button('预处理')
                with gr.Column():
                    with gr.Row():
                        aligned_face = gr.Image(label='获取人像',
                                                type='numpy',
                                                interactive=False)
                with gr.Column():
                    reconstructed_face = gr.Image(label='人脸重建',
                                                  type='numpy',
                                                  interactive=False)
                    instyle = gr.Variable()

            # 图像选择示例
            # with gr.Row():
            #     paths = sorted(pathlib.Path('images').glob('*.jpg'))
            #     example_images = gr.Dataset(components=[input_image],
            #                                 samples=[[path.as_posix()]
            #                                          for path in paths])                    

        with gr.Box():
            with gr.Row():
                with gr.Column():
                    # 显示油画图像选择预览图
                    # text = get_style_image_markdown_text('cartoon')
                    # style_image = gr.Markdown(value=text)
                    style_index = gr.Slider(0,
                                            316,
                                            value=26,
                                            step=1,
                                            label='风格图像序号')
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        structure_weight = gr.Slider(0,
                                                     1,
                                                     value=0.6,
                                                     step=0.1,
                                                     label='Structure Weight')
                    with gr.Row():
                        color_weight = gr.Slider(0,
                                                 1,
                                                 value=1,
                                                 step=0.1,
                                                 label='Color Weight')
                    with gr.Row():
                        structure_only = gr.Checkbox(label='Structure Only')
                    with gr.Row():
                        generate_button = gr.Button('开始生成')

                with gr.Column():
                    result = gr.Image(label='结果图',interactive=False)


        # 预处理按钮点击执行
        # preprocess_button.click(fn=model.detect_and_align_face,
        #                         inputs=[input_image],
        #                         outputs=aligned_face)

        # aligned_face.change(fn=model.reconstruct_face,
        #                     inputs=[aligned_face, encoder_type],
        #                     outputs=[
        #                         reconstructed_face,
        #                         instyle,
        #                     ])
        # style_type.change(fn=update_slider,
        #                   inputs=style_type,
        #                   outputs=style_index)
        # style_type.change(fn=update_style_image,
        #                   inputs=style_type,
        #                   outputs=style_image)

        # 生成风格图像按钮点击执行
        # generate_button.click(fn=model.generate,
        #                       inputs=[
        #                           style_type,
        #                           style_index,
        #                           structure_weight,
        #                           color_weight,
        #                           structure_only,
        #                           instyle,
        #                       ],
        #                       outputs=result)
        # example_images.click(fn=set_example_image,
        #                      inputs=example_images,
        #                      outputs=example_images.components)
        # example_styles.click(fn=set_example_styles,
        #                      inputs=example_styles,
        #                      outputs=example_styles.components)
        # example_weights.click(fn=set_example_weights,
        #                       inputs=example_weights,
        #                       outputs=example_weights.components)

    # 启动webui
    demo.launch(
        # enable_queue=args.enable_queue,
        # server_port=args.port,
        # share=args.share,
    )


if __name__ == '__main__':
    main()         