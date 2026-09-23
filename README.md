# 基于生成对抗网络的老旧照片修复系统

本项目是智能科学与技术专业毕业设计的基础工程，用于实现老旧照片的上传、修复、预览与下载。

## 当前功能

- 本地桌面图形界面
- 图片上传与原图/结果对比
- GFPGAN V1.4 人脸修复开关
- Real-ESRGAN X4 图像清晰化开关，支持 2 倍或 4 倍输出
- 模型权重与 Python 推理依赖状态提示

## 项目结构

```text
old_photo_restoration_system/
├── app.py                  # 程序入口
├── app/                    # 界面相关代码
├── services/               # 修复流程与模型调用
├── models/                 # 预训练模型权重（不提交到 Git）
├── assets/                 # 页面静态资源
├── data/input/             # 待修复图片
├── data/output/            # 修复结果
└── tests/                  # 测试代码
```

## 模型权重位置

```text
models/
├── gfpgan/GFPGANv1.4.pth
└── realesrgan/RealESRGAN_x4plus.pth
```

GFPGAN 首次进行未对齐人脸修复时还需要 FaceXLib 的检测和人脸解析权重，已放置在：

```text
gfpgan/weights/
├── detection_Resnet50_Final.pth
└── parsing_parsenet.pth
```

## 快速启动

```powershell
conda activate D:\conda_envs\old-photo-restoration
cd C:\Users\96551\Desktop\old_photo_restoration_system
python app.py
```

程序将打开本地桌面窗口。

在 PyCharm 中，将项目解释器设为：

```text
D:\conda_envs\old-photo-restoration\python.exe
```

## 后续计划

1. 接入 Microsoft 老照片整体修复模型，增加去划痕阶段。
2. 增加可选的黑白照片上色模块。
3. 保存修复记录，并补充效果评价与对比实验。
