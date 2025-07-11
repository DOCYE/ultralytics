# 这个一定要加，否则总是报错,即使检测到同一个进程中加载了多个 OpenMP 运行时库的副本，也请允许这种情况发生，不要报错或终止进程。
# 如果不加，会出现爆显存的问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# 确保所有创建子进程的代码（如model.train()）都在这个块内
if __name__ == '__main__':
    # Create a new YOLO model from scratch (这行会被下一行覆盖，可以删除)
    # model = YOLO("yolo11n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data="coco8.yaml", epochs=3)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model("https://ultralytics.com/images/bus.jpg") # 完整的URL

    # Export the model to ONNX format
    success = model.export(format="onnx")

    print("模型训练、验证、推理和导出完成！")




