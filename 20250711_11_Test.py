# 这个一定要加，否则总是报错,即使检测到同一个进程中加载了多个 OpenMP 运行时库的副本，也请允许这种情况发生，不要报错或终止进程。
# 如果不加，会出现爆显存的问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# from ultralytics import settings
# # 修改数据集目录 (这部分可以在if __name__ == '__main__': 外部，因为它不直接创建子进程)
# settings.update({"datasets_dir": r"C:\Users\yemin\PycharmProjects\ultralytics\datasets"})

from ultralytics import YOLO

# 确保所有创建子进程的代码（如model.train()）都在这个块内
if __name__ == '__main__':
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    train_results = model.train(data="coco8.yaml", epochs=3)
    print("Training Results:", train_results)  # 打印训练结果

    # Evaluate the model's performance on the validation set
    val_results = model.val()
    print("Validation Results:", val_results)  # 打印验证结果

    # Perform object detection on an image using the model
    # detect_results = model("https://ultralytics.com/images/bus.jpg")  # 完整的URL
    detect_results = model(r"C:\Users\yemin\PycharmProjects\ultralytics\20250711_test01.jpg")

    # 打印检测结果
    print("Detection Results:")
    print(detect_results)  # 打印整个结果对象以检查其结构

    # 访问检测到的对象信息
    for result in detect_results:
        # 检查是否有 boxes 属性
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                class_id = box.cls.item()  # 类别ID
                confidence = box.conf.item()  # 置信度
                bbox = box.xyxy[0].tolist()  # 边界框坐标

                print(f"Detected: {class_id} with confidence {confidence}")
                print(f"Bounding box: {bbox}")
        else:
            print("No boxes found in this result.")




