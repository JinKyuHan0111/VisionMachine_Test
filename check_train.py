import csv
import time
import os

while True:
    os.system('cls')
    try:
        with open('runs/train/fire_detector/results.csv') as f:
            rows = list(csv.DictReader(f))
        r = rows[-1]
        epoch = int(float(r['epoch']))
        map50 = float(r['metrics/mAP50(B)'])
        map50_95 = float(r['metrics/mAP50-95(B)'])
        train_loss = float(r['train/box_loss'])
        val_loss = float(r['val/box_loss'])

        print(f'에폭        : {epoch}/100')
        print(f'mAP@0.5     : {map50:.4f}')
        print(f'mAP@0.5:0.95: {map50_95:.4f}')
        print(f'train_loss  : {train_loss:.4f}')
        print(f'val_loss    : {val_loss:.4f}')
    except Exception as e:
        print(f'대기 중... ({e})')

    time.sleep(30)
