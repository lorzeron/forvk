from ultralytics import YOLO
from PIL import Image

# Используем Yolo v5 для детекции объектов, простая модель, уже предобученная
model = YOLO('yolov5s.pt') 

# Основная функция
def logo_search(image_path):
    # Загрузка изображения
    image = Image.open(image_path)
    results = model.predict(source=image_path, save=True, save_txt=True)
    print(f"Готово: {results}")
    results[0].plot() # Выводим результаты на изображении

if __name__ == "__main__":
    test_image = "файл.jpg"  # Путь к изображению
    logo_search(test_image)