from ultralytics import YOLO

def main():
    # 1. Load the pre-trained base model
    model = YOLO('yolov8n.pt') 

    # 2. Train using your 4GB GPU with workers=0 to fix Windows multiprocessing
    model.train(
        data='datasets/sard/data.yaml', 
        epochs=50,                      
        imgsz=640,                      
        batch=4,          # Safe batch size for 4GB VRAM
        device=0,         # Force CUDA GPU
        workers=0,        # <-- Disables multi-threading on Windows to prevent the crash
        project='NIDAR_Vision',
        name='sard_finetune_v1'
    )

if __name__ == '__main__':
    main()