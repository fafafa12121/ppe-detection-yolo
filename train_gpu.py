import os
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    print(f'CUDA可用: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')

    # 自动定位数据集
    BASE = r'E:\shuju'
    found = None
    for root_dir, dirs, files in os.walk(BASE):
        if 'data.yaml' in files:
            found = root_dir
            break
    if not found:
        print('❌ 未找到数据集，请检查E:\\shuju')
        exit()
    print(f'✅ 数据集: {found}')

    # 确保配置文件存在
    yaml_path = os.path.join(found, 'ppe_data.yaml')
    if not os.path.exists(yaml_path):
        import yaml
        data = {'path': found.replace('\\', '/'),
                'train': 'train/images',
                'val': 'valid/images',
                'nc': 4,
                'names': ['Helmet', 'NoHelmet', 'NoVest', 'Vest']}
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

    # 方案A：稳定加速
    model = YOLO('yolov8n.pt')
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,        # 降低显存压力
        device=0,
        amp=False,      # 之前已关闭，保持
        workers=2,      # 并行加载数据，减少GPU等待
        name='ppe_gpu_fast'
    )