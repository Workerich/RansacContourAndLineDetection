import cv2
import numpy as np
import time


def mnk_line(points):
    """
    МНК для прямой y = k*x + b
    points: список кортежей (x, y)
    возвращает (k, b) и название модели
    """
    if len(points) < 2:
        return None, None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    n = len(points)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-6:
        return None, None

    k = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - k * sum_x) / n

    return (k, b), "Прямая"


def mnk_circle(points):
    """
    МНК для окружности (x - x0)^2 + (y - y0)^2 = R^2
    points: список кортежей (x, y)
    возвращает (x0, y0, R) и название модели
    """
    if len(points) < 3:
        return None, None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    A = np.vstack([x, y, np.ones(len(points))]).T
    b = -(x ** 2 + y ** 2)

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    A_coef, B_coef, C_coef = params

    x0 = -A_coef / 2
    y0 = -B_coef / 2
    R = np.sqrt(max(0, x0 ** 2 + y0 ** 2 - C_coef))

    return (x0, y0, R), "Окружность"


def mnk_parabola(points):
    """
    МНК для параболы y = a*x^2 + b*x + c
    points: список кортежей (x, y)
    возвращает (a, b, c) и название модели
    """
    if len(points) < 3:
        return None, None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    A = np.vstack([x ** 2, x, np.ones(len(points))]).T
    params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = params

    return (a, b, c), "Парабола"


def calculate_rmse(points, model, model_type):
    """
    Вычисляет RMSE (среднеквадратичную ошибку) для модели
    """
    if len(points) == 0:
        return float('inf')

    errors = []

    for x, y in points:
        if model_type == "Прямая":
            k, b = model
            y_pred = k * x + b
            error = y - y_pred

        elif model_type == "Окружность":
            x0, y0, R = model
            dist_to_center = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            error = dist_to_center - R

        elif model_type == "Парабола":
            a, b, c = model
            y_pred = a * x ** 2 + b * x + c
            error = y - y_pred

        else:
            return float('inf')

        errors.append(error ** 2)

    mse = np.mean(errors)
    rmse = np.sqrt(mse)

    return rmse


def select_best_model(points):
    """
    Пробует все три модели и выбирает лучшую по RMSE
    возвращает (model, model_name, rmse, all_results)
    """
    models = [
        ('Прямая', mnk_line),
        ('Окружность', mnk_circle),
        ('Парабола', mnk_parabola)
    ]

    best_model = None
    best_name = None
    best_rmse = float('inf')
    all_results = []

    for name, func in models:
        model, _ = func(points)

        if model is None:
            print(f"  {name}: не удалось построить (мало точек)")
            continue

        rmse = calculate_rmse(points, model, name)
        all_results.append({
            'name': name,
            'model': model,
            'rmse': rmse
        })

        print(f"  {name}: RMSE = {rmse:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    return best_model, best_name, best_rmse, all_results


def draw_line(image, model, color=(0, 0, 255), thickness=2):
    """Рисует прямую на изображении"""
    k, b = model
    h, w = image.shape[:2]

    x1 = 0
    y1 = int(k * x1 + b)
    x2 = w
    y2 = int(k * x2 + b)

    cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def draw_circle(image, model, color=(0, 0, 255), thickness=2):
    """Рисует окружность на изображении"""
    x0, y0, R = model
    cv2.circle(image, (int(x0), int(y0)), int(R), color, thickness)


def draw_parabola(image, model, color=(0, 0, 255), thickness=2):
    """Рисует параболу на изображении"""
    a, b, c = model
    h, w = image.shape[:2]

    points = []
    for x in range(0, w, 2):  # шаг 2 пикселя для производительности
        y = int(a * x ** 2 + b * x + c)
        if 0 <= y < h:
            points.append((x, y))

    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], color, thickness)


def draw_model(image, model, model_name, color=(0, 0, 255), thickness=2):
    """
    Универсальная функция отрисовки модели
    """
    if model_name == "Прямая":
        draw_line(image, model, color, thickness)
    elif model_name == "Окружность":
        draw_circle(image, model, color, thickness)
    elif model_name == "Парабола":
        draw_parabola(image, model, color, thickness)
    else:
        print(f"Неизвестная модель: {model_name}")

def fit_and_draw(points, img_color):
    """
    Принимает точки и изображение, строит аппроксимации,
    возвращает изображение с нарисованной лучшей моделью,
    название модели и точки
    """
    if points is None or len(points) < 3:
        print("Недостаточно точек для аппроксимации")
        return img_color, None, None

    print(f"\nВсего точек: {len(points)}")
    print("Аппроксимация моделей:")

    # Выбираем лучшую модель
    created = time.time()
    best_model, best_name, best_rmse, all_results = select_best_model(points)
    ended = time.time()

    if best_model is None:
        msg = "Не удалось построить ни одну модель"
        return img_color, None, msg

    # Копируем изображение для отрисовки
    result_img = img_color.copy()

    # Рисуем лучшую модель красным цветом
    draw_model(result_img, best_model, best_name, color=(0, 0, 255), thickness=1)

    msg = f"Возможно - {best_name}\nВремя {round(ended - created, 2)} сек"
    return result_img, best_name, msg


# ========== Пример использования с вашей функцией ==========

def get_points_from_image(img_color, threshold=80, bininv=False):
    """
    Ваша функция для получения точек
    """
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    lower = (0, 0, 0)
    upper = (179, 255, threshold)

    img_binary = cv2.inRange(img_hsv, lower, upper)

    if bininv:
        img_binary = cv2.bitwise_not(img_binary)

    points = np.column_stack(np.where(img_binary > 0))
    points = [(int(x), int(y)) for y, x in points]

    if len(points) < 3:
        print("Недостаточно точек")
        return None

    print(f"Найдено точек: {len(points)}")
    return points

def process_mnk(image_path, threshold=100, bininv=False):
    """
    Полный процесс: загрузка -> поиск точек -> аппроксимация -> результат
    """
    # Загружаем изображение
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None, None

    # Получаем точки
    points = get_points_from_image(img_color, threshold=threshold, bininv=bininv)

    if points is None:
        return img_color, None

    # Строим аппроксимацию и рисуем
    result_img, best_name, msg = fit_and_draw(points, img_color)

    return msg, result_img
