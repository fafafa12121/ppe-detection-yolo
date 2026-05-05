import os
from pathlib import Path
from ultralytics import YOLO

# ========== 自动找数据集 ==========
BASE = r'E:\shuju'
found = None
for root_dir, dirs, files in os.walk(BASE):
    if 'data.yaml' in files:
        found = root_dir
        break

if not found:
    print('❌ 没找到数据集，请检查E:\\shuju')
    exit()

print(f'✅ 数据集位置: {found}')

# 快速检查
for split in ['train', 'valid']:
    img_dir = os.path.join(found, split, 'images')
    lbl_dir = os.path.join(found, split, 'labels')
    imgs = len(os.listdir(img_dir))
    lbls = len(os.listdir(lbl_dir))
    print(f'{split}: {imgs}张图, {lbls}个标签', '✓' if imgs==lbls else '✗')

# ========== 训练参数 ==========
EPOCHS = 2     # <-- 想改轮数就改这里
BATCH = 8

# 确保有ppe_data.yaml（没有就生成）
yaml_path = os.path.join(found, 'ppe_data.yaml')
if not os.path.exists(yaml_path):
    import yaml
    data = {'path': found.replace('\\','/'), 'train': 'train/images', 'val': 'valid/images',
            'nc': 4, 'names': ['Helmet','NoHelmet','NoVest','Vest']}
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

model = YOLO('yolov8n.pt')
model.train(data=yaml_path, epochs=EPOCHS, imgsz=640, batch=BATCH, name='ppe_fast')