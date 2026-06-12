import numpy as np
import time
import random
import math
import cv2


def get_points_from_image(img_color, threshold=80, bininv=False):
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

    print(len(points))
    return points


# DISTANCES
def dist_point_to_point(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def dist_to_line(p, a, b, c):
    return abs(a * p[0] + b * p[1] + c) / (a * a + b * b) ** 0.5

def dist_to_circle(p, cx, cy, r):
    x, y = p
    d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return abs(d - r)

def dist_to_parabola(p, a, b, c):
    x, y = p
    y_model = a * x**2 + b * x + c
    return abs(y - y_model)


# PARAMETERS
def get_line_params(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return a, b, c

def get_circle_params(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if abs(d) < 1e-6:
        return None

    cx = (
        (x1 ** 2 + y1 ** 2) * (y2 - y3) +
        (x2 ** 2 + y2 ** 2) * (y3 - y1) +
        (x3 ** 2 + y3 ** 2) * (y1 - y2)
    ) / d

    cy = (
        (x1 ** 2 + y1 ** 2) * (x3 - x2) +
        (x2 ** 2 + y2 ** 2) * (x1 - x3) +
        (x3 ** 2 + y3 ** 2) * (x2 - x1)
    ) / d

    r = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)

    return cx, cy, r

def get_parabola_params(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = (
        (x1 - x2) *
        (x1 - x3) *
        (x2 - x3)
    )

    if abs(d) < 1e-6:
        return None

    a = (
        x3 * (y2 - y1) +
        x2 * (y1 - y3) +
        x1 * (y3 - y2)
    ) / d

    b = (
        x3**2 * (y1 - y2) +
        x2**2 * (y3 - y1) +
        x1**2 * (y2 - y3)
    ) / d

    c = (
        x2 * x3 * (x2 - x3) * y1 +
        x3 * x1 * (x3 - x1) * y2 +
        x1 * x2 * (x1 - x2) * y3
    ) / d

    return a, b, c


# CLEANING
def clean_res(arr, E=5):
    E = E * 2 - 1
    clusters = [[arr[0]]]
    for i in range(len(arr) - 1):
        x1, y1 = arr[i]
        x2, y2 = arr[i + 1]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist < E:
            clusters[-1].append(arr[i + 1])
        else:
            clusters.append([arr[i + 1]])

    mx = -1
    idx_mx = -1
    for i in range(len(clusters)):
        if mx < len(clusters[i]):
            mx = len(clusters[i])
            idx_mx = i

    return clusters[idx_mx]


# RANSAC
def get_point_projection(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    c = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
    z = (x2 - x1) ** 2 + (y2 - y1) ** 2

    return c / z

def ransac_line_segment(arr, iterations=50, E=5):
    best_points = []
    max_inliers = -1

    for _ in range(iterations):
        idx = random.sample(range(len(arr)), 2)
        p1, p2 = arr[idx[0]], arr[idx[1]]
        a, b, c = get_line_params(p1, p2)

        inliers = []
        for p in arr:
            t = get_point_projection(p, p1, p2)
            if t <= 0:
                d = dist_point_to_point(p, p1)
            elif t >= 1:
                d = dist_point_to_point(p, p2)
            else:
                d = dist_to_line(p, a, b, c)

            if d <= E:
                inliers.append(p)

        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_points = inliers

    return best_points

def ransac_line(arr, iterations=50, E=5):
    best_points = []
    max_inliers = -1

    for _ in range(iterations):
        idx = random.sample(range(len(arr)), 2)
        p1, p2 = arr[idx[0]], arr[idx[1]]

        if p1[0] == p2[0]:
            inliers = [p for p in arr if abs(p[0] - p1[0]) < E]
        else:
            a, b, c = get_line_params(p1, p2)
            inliers = [p for p in arr if dist_to_line(p, a, b, c) < E]

        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_points = inliers

    return best_points


def ransac_circle(arr, iterations=100, E=5):
    best_points = []
    max_inliers = -1

    for _ in range(iterations):
        idx = random.sample(range(len(arr)), 3)

        p1 = arr[idx[0]]
        p2 = arr[idx[1]]
        p3 = arr[idx[2]]

        params = get_circle_params(p1, p2, p3)

        if params is None:
            continue

        cx, cy, r = params

        inliers = [
            p for p in arr
            if dist_to_circle(p, cx, cy, r) < E
        ]

        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_points = inliers

    return best_points

def ransac_parabola(arr, iterations=100, E=5):
    best_points = []
    max_inliers = -1

    for _ in range(iterations):

        idx = random.sample(range(len(arr)), 3)

        p1 = arr[idx[0]]
        p2 = arr[idx[1]]
        p3 = arr[idx[2]]

        params = get_parabola_params(p1, p2, p3)

        if params is None:
            continue

        a, b, c = params

        inliers = [
            p for p in arr
            if dist_to_parabola(p, a, b, c) < E
        ]

        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_points = inliers

    return best_points


# OTHER
def find_contours(arr, ransac_model, iterations, w, E, min_length, max_lines, filtered=True):
    all_lines = []
    remaining = arr
    anomalies = int(len(arr) * (1 - w) / max_lines)
    k = iterations
    print(anomalies, "Теоретическое кол-во выбросов")

    for _ in range(max_lines):
        if len(remaining) <= anomalies:
            break

        best_line = ransac_model(remaining, k, E)
        if filtered:
            best_line = clean_res(sorted(best_line), E)

        if len(best_line) < min_length:
            break
        remaining = [p for p in remaining if p not in best_line]
        all_lines.append(best_line)
        k += 1

    return list(all_lines)


ransac_models = {
    "line": (ransac_line_segment, 2, "прямая"),
    "circle": (ransac_circle, 3, "окружность"),
    "parabola": (ransac_parabola, 3, "парабола")
}

def process_ransac(path, model_type, find_contour, w, E, min_length, max_contours, filtered):
    img = cv2.imread(path)
    arr = get_points_from_image(img, threshold=100)
    ransac_model, min_num_points, model_name = ransac_models[model_type]
    k = math.ceil(math.log(1 - 0.99) / math.log(1 - w ** min_num_points))
    print(k, "Итераций")

    if find_contour:
        created = time.time()
        good_contours = find_contours(arr, ransac_model, k, w, E, min_length,
                                      max_contours, filtered)
        ended = time.time()
        good_points = set()
        for line in good_contours:
            good_points.update(line)
        msg = (f"Кол-во найденных совпадений: {len(good_contours)}\n"
               f"Модель - {model_name}\n"
               f"Время: {round(ended - created, 2)} cек")

    else:
        created = time.time()
        good_points = ransac_model(arr, k, E)
        ended = time.time()
        msg = (f"Модель - {model_name}\n"
               f"Время: {round(ended - created, 2)} cек")

    for p in good_points:
        img[p[1], p[0]] = (0, 0, 255)

    return msg, img
