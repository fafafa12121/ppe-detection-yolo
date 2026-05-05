import os, yaml, sys
from pathlib import Path
from ultralytics import YOLO

# ================= 自动查找数据集路径 =================
BASE = r'E:\shuju'
FOUND = None
for root_dir, dirs, files in os.walk(BASE):
    if 'data.yaml' in files:
        FOUND = root_dir
        break

if not FOUND:
    print('❌ 在 E:\\shujuji 下没找到 data.yaml，请确认数据集已解压到此目录。')
    sys.exit(1)

root = FOUND
print(f'✅ 自动定位到数据集: {root}')
# =====================================================

PYTHON_EXE = r'E:\Python\python\python.exe'
EPOCHS = 2
BATCH = 8

def generate_yaml():
    yaml_path = os.path.join(root, 'ppe_data.yaml')
    if not os.path.exists(yaml_path):
        data = {
            'path': root.replace('\\', '/'),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': 4,
            'names': ['Helmet', 'NoHelmet', 'NoVest', 'Vest']
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
        print("✅ ppe_data.yaml 已创建")
    else:
        print("✅ ppe_data.yaml 已存在")

def data_check():
    print("\n--- 数据完整性检查 ---")
    for split in ['train', 'valid']:
        img_dir = os.path.join(root, split, 'images')
        lbl_dir = os.path.join(root, split, 'labels')
        imgs = os.listdir(img_dir) if os.path.exists(img_dir) else []
        lbls = os.listdir(lbl_dir) if os.path.exists(lbl_dir) else []
        img_stems = {Path(f).stem for f in imgs}
        lbl_stems = {Path(f).stem for f in lbls}
        print(f"{split}: {len(imgs)}张图, {len(lbls)}个标签", end='')
        if not (img_stems - lbl_stems) and not (lbl_stems - img_stems):
            print(" ✓ 全部匹配")
        else:
            print(f" ✗ 不匹配")

def data_augment():
    print("\n--- 快速数据增强（前300张） ---")
    import albumentations as A
    import cv2
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    src_img_dir = os.path.join(root, 'train', 'images')
    src_lbl_dir = os.path.join(root, 'train', 'labels')
    img_files = sorted(os.listdir(src_img_dir))[:300]
    for i, fname in enumerate(img_files):
        img_path = os.path.join(src_img_dir, fname)
        lbl_path = os.path.join(src_lbl_dir, Path(fname).stem + '.txt')
        img = cv2.imread(img_path)
        if img is None: continue
        bboxes, class_labels = [], []
        with open(lbl_path) as f:
            for line in f.readlines():
                p = line.strip().split()
                class_labels.append(int(p[0]))
                bboxes.append([float(x) for x in p[1:]])
        try:
            aug = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            new_fname = f"{Path(fname).stem}_aug.jpg"
            cv2.imwrite(os.path.join(src_img_dir, new_fname), aug['image'])
            with open(os.path.join(src_lbl_dir, Path(new_fname).stem + '.txt'), 'w') as f:
                for cls, bbox in zip(aug['class_labels'], aug['bboxes']):
                    f.write(f"{cls} {' '.join([f'{x:.6f}' for x in bbox])}\n")
        except: pass
        if i % 50 == 0: print(f"  已处理 {i}/{len(img_files)}")
    print("✅ 增强完成")

def train():
    print(f"\n--- 开始训练 ({EPOCHS}轮, batch={BATCH}) ---")
    model = YOLO('yolov8n.pt')
    model.train(data=os.path.join(root, 'ppe_data.yaml'),
                epochs=EPOCHS, imgsz=640, batch=BATCH, name='ppe_final')

if __name__ == '__main__':
    print("="*50)
    print("  PPE安全装备检测 — 自动适配版")
    print("="*50)
    generate_yaml()
    data_check()
    data_augment()
    train()