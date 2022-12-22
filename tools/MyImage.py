from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
'''
这个文件中包含了一些图像的工具方法：

一、图像：
1、替换指定像素BGR值
2、图片转视频
3、获取指定位置BGR值或RGB值

3、从视频中提取出图片
4、将视频转换格式
'''


class MyImageUtils:
    '''
    此类是一个处理图像的工具类：
    传入__init__的参数如下：
    path：如果为文件，则为图像的文件根目录；否则为一张图像。
    '''

    def __init__(self, path):
        self.path = path
        if not os.path.isdir(self.path):
            self.img = cv2.imread(path)

    def get_BRG(self, img=None, typeChannel="RGB"):
        '''
        这是用于提取图片中指定位置像素BGR值的方法：
        img：图片实例化对象，若为None则使用传入self.img
        typeChannel：
        '''

        ix = 0
        iy = 0
        b = 0
        g = 0
        r = 0
        def draw_rectangle(event, x, y, flag, param):
            nonlocal ix, iy, b, g, r
            if event==cv2.EVENT_LBUTTONDOWN :
                print(f"the ({str(y)} , {str(x)} ) BRG is {img[y, x]}")
                ix = x      
                iy = y      
                b, g, r = img[y, x][:3]       
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_rectangle)
        #显示并延时
        if img is None:
            img = self.img
        while(1):
            cv2.imshow('image',img)
            k = cv2.waitKey(20)
            if k == 27 or k == ord(' '):
                break
        #销毁所有窗口
        cv2.destroyAllWindows()
        if typeChannel == "RGB" or "rgb":
            return (r, g, b)
        return (b, g, r)

    def replace_BGR(self, inputBRG=(235, 235, 235), outputBRG=(255, 255, 255), showImage=False, replace=True):

        '''
        这是用于替换指定颜色的像素更改成其他颜色的方法，传入参数：
        imgPath:       图片路径
        inputBRG:  输入的BRG值，当传入的值为None时，将会显示图片让用户通过鼠标指定水印的RGB。
        outputBGR: 输出的BGR值。
        showImage: 选择是否显示输出图像。
        replace: 选择是否替换原图像。
        '''

        def rp(imgPath, inputBRG, showImage=False, replace=True):
            img_nd = cv2.imread(imgPath)
            if inputBRG is None:
                inputBRG = self.get_BRG(img_nd)
            channels = img_nd.shape[2]
            height = img_nd.shape[0]
            weight = img_nd.shape[1]
            for x in range(height):
                for y in range(weight):
                    if channels == 3:
                        (r, g, b) = img_nd[x, y]
                    else:
                        (r, g, b, a) = img_nd[x, y]
                    if ((r >= inputBRG[0] - 15 and r <= inputBRG[0] + 15) and 
                        (g >= inputBRG[1] - 15 and g <= inputBRG[1] + 15) and 
                        (b >= inputBRG[2] - 15 and b <= inputBRG[2] + 15)) :
                        
                        img_nd[x, y][:3] = outputBRG
            img_end = Image.fromarray(img_nd)
            if showImage:
                img_end.show()
            if replace:
                img_end.save(imgPath)
        if os.path.isdir(self.path):
            for dir_path, dirnames, filenames in os.walk(self.path):
                for filename in filenames:
                    if filename.split('.')[-1] in ["jpg", "png"]:
                        rp(str(os.path.join(dir_path, filename)), inputBRG, showImage, replace)
        else:
            rp(self.path, inputBRG, showImage, replace)


    def imgs_to_video(self, fps, showVideo=False, saveVideo=False, videoName="output.mp4"):

        '''
        本方法适用于将某文件夹中的所有图片通过逐帧录入的方式得到视频，传入参数：
        imgsRoot: 图片文件夹路径
        fps：转换后视频帧数
        showVideo：显示视频
        saveVideo：是否保存视频文件
        videoName：视频文件名
        ****
        以下是一些fourcc编码格式：
        cv2.VideoWriter_fourcc('P','I','M','1') = MPEG-1 codec
        cv2.VideoWriter_fourcc('M','J','P','G') = motion-jpeg codec --> mp4v
        cv2.VideoWriter_fourcc('M', 'P', '4', '2') = MPEG-4.2 codec
        cv2.VideoWriter_fourcc('D', 'I', 'V', '3') = MPEG-4.3 codec
        cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') = MPEG-4 codec --> avi
        cv2.VideoWriter_fourcc('U', '2', '6', '3') = H263 codec
        cv2.VideoWriter_fourcc('I', '2', '6', '3') = H263I codec
        cv2.VideoWriter_fourcc('F', 'L', 'V', '1') = FLV1 codec
        '''

        codec = {'mp4':'mp4v', 'avi':'XVID', 'flv':'flv1'}
        fourcc = cv2.VideoWriter_fourcc(*codec[videoName.split('.')[-1]])
        videoWriter = None
        imgsRoot = self.path
        for imgFile in os.listdir(imgsRoot):
            imgPath = os.path.join(imgsRoot, imgFile)
            if imgFile.split('.')[-1] not in ["jpg", "png"]:
                continue
            frame = cv2.imread(imgPath)
            if saveVideo:
                if videoWriter is None:
                    imgSize = frame.shape[-2:-4:-1]
                    videoWriter = cv2.VideoWriter(os.path.join(imgsRoot, videoName), fourcc, fps, imgSize, True)
                videoWriter.write(frame)
                print(imgPath, " end!")
            if showVideo:
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) and 0xFF == ord('q'):
                    break
        if videoWriter is  not None:
            videoWriter.release()
        cv2.destroyAllWindows()

class MyVideoUtils:
    '''
    此类是一个用于对视频进行处理的类
    传入__init__的参数为：
    path: 视频文件路径
    '''

    def __init__(self, path):
        self.path = path
        self.videoCap = cv2.VideoCapture(self.path)
        self.fps = self.videoCap.get(cv2.CAP_PROP_FPS)
        self.totalFrame = self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.width = self.videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        isOpen = self.videoCap.isOpened
        assert isOpen, "Can't open video!"
        
    def video_to_imgs(self, imgsPath, intervel=1):
        '''
        本方法用于从视频中提取出图片放到指定文件夹， 参数如下：
        videoPath: 视频地址
        imgsPath： 用于放图片的文件夹，若没有则自动创建该文件夹
        intervel: 间隔多少帧提取图片
        '''
        
        os.makedirs(imgsPath, exist_ok=True)
        videoCap = self.videoCap
        for i in range(int(self.totalFrame)):
            (_, frame) = videoCap.read()
            if i % intervel == 0:
                imgPath = os.path.join(imgsPath, str(i) + ".jpg")
                cv2.imwrite(imgPath, frame, [cv2.IMWRITE_JPEG_QUALITY,100])
                print(imgPath, " end!")

    def change_video_format(self, targetFormat, targetPath=None):
        '''
        此方法用于将视频的格式转换到指定格式，传入参数如下：
        targetFormat: 指定的视频格式 例如："mp4","flv"等等
        overwrite: 是否覆盖原文件？默认为False
        targetPath: 输出视频保存地址，若传入的是文件路径，则直接保存至该路径；若为文件夹路径，则保存至该文件夹下文件名称为output的文件。
        '''
        
        codec = {'mp4':'mp4v', 'avi':'XVID', 'flv':'flv1'}
        assert targetFormat in codec.keys(), f"Can't change into {targetFormat} file"
        fourcc = cv2.VideoWriter_fourcc(*codec[targetFormat])
        videoWriter = None
        
        if targetPath is None:
            targetPath = self.path.split(".")[0] + "." + targetFormat
        if os.path.isdir(targetPath):
            targetPath = os.path.join(targetPath, "output." + targetFormat)
        for i in range(int(self.totalFrame)):
            (_, frame) = self.videoCap.read()
            if videoWriter is None:
                imgSize = frame.shape[-2:-4:-1]
                videoWriter = cv2.VideoWriter(targetPath, fourcc, self.fps, imgSize, True)
            videoWriter.write(frame)
        if videoWriter is  not None:
            videoWriter.release()
        cv2.destroyAllWindows()

