import numpy as np
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

def dist_to_model(p, k, b):
    return abs(k * p[0] - p[1] + b) / (k ** 2 + 1) ** 0.5

def get_line_params(p1, p2):
    x = np.array([p1[0], p2[0]])
    y = np.array([p1[1], p2[1]])
    k, b = np.polyfit(x, y, 1)
    return k, b

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

def ransac(arr, iterations=50, E=5):
    best_points = []
    max_inliers = -1

    for _ in range(iterations):
        idx = random.sample(range(len(arr)), 2)
        p1, p2 = arr[idx[0]], arr[idx[1]]

        if p1[0] == p2[0]:
            inliers = [p for p in arr if abs(p[0] - p1[0]) < E]
        else:
            k, b = get_line_params(p1, p2)
            inliers = [p for p in arr if dist_to_model(p, k, b) < E]

        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            best_points = inliers

    return best_points

def find_contour(arr, w, E, min_length, max_lines, filtered=True):
    all_lines = []
    remaining = arr
    k = int(math.log(1 - 0.99) / math.log(1 - w**2))
    print(k, "Итераций")
    anomalies = int(len(arr) * (1 - w) / max_lines)
    print(anomalies, "Теоретическое кол-во выбросов")

    for _ in range(max_lines):
        if len(remaining) <= anomalies:
            break

        best_line = ransac(remaining, k, E)
        if filtered:
            best_line = clean_res(sorted(best_line), E)

        if len(best_line) < min_length:
            break
        remaining = [p for p in remaining if p not in best_line]
        all_lines.append(best_line)
        k += 1

    return list(all_lines)

def process_ransac(path, w, E, min_length, max_lines, filtered):
    img = cv2.imread(path)
    arr = get_points_from_image(img, threshold=100)
    good_lines = find_contour(arr, w, E, min_length, max_lines, filtered)
    for line in good_lines:
        for p in line:
            img[p[1], p[0]] = (0, 0, 255)

    return f"Кол-во прямых: {len(good_lines)}", img