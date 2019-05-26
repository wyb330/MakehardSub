import cv2
import numpy as np
from matplotlib import pyplot as plt

BINARY_THREHOLD = 180


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(img, morpholog=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising
    img = cv2.fastNlMeansDenoising(img, h=10, searchWindowSize=21, templateWindowSize=7)
    # 적응임계처리를 이용한 이미지 흑백 처리
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

    # 이미지의 노이즈나 hole를 제거
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    img = image_smoothening(img)
    if morpholog:
        img = cv2.bitwise_or(img, closing)
    return img


def contour_plot_on_text_in_image(img):
    # 이미지 상에 있는 segmenent들의 경계를 부드럽게 하고 구멍을 메꿔준다.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 3))
    dilated = cv2.dilate(img, kernel, iterations=3)

    # 이미지 상에 있는 영역의 윤곽선 목록을 추출한다.
    _, contours, hierarchy = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)  # get contours

    return contours, dilated


def min_area_rect(rects):
    points = np.array(rects)
    x = np.min(points[:, 0])
    x_m = np.max(points[:, 0])
    y = np.min(points[:, 1])
    y_m = np.max(points[:, 1])
    w = np.max(points[:, 2])
    h = np.max(points[:, 3])

    return int(x), int(y), int(x_m - x + w), int(y_m - y + h)


def cluster_contour(rects):

    # line에서 x축 거리가 일정 거리 이상 떨어진 영역을 별도로 분리한다.
    def cluster_x(points):
        points = sorted(points, key=lambda x: x[0])
        groups = []
        g = []
        for idx, p in enumerate(points):
            if idx < len(points) - 1:
                p_next = points[idx + 1]
                if abs(p_next[0] - (p[0] + p[2])) > 50:
                    g.append(p)
                    groups.append(g)
                    g = []
                else:
                    g.append(p)
            else:
                g.append(p)
                groups.append(g)

        return groups

    def cluster_y(points):
        g = []
        for idx, p in enumerate(points):
            if idx < len(points) - 1:
                p_next = points[idx + 1]
                if (abs(p[1] - (p_next[1] + p_next[3])) < 40) and (abs(p[0] - p_next[0]) < 35):
                    g.append(p)
                    g.append(p_next)
                else:
                    if len(g) > 0:
                        break
        return g

    y = dict()
    lines = [r[1] + r[3] for r in rects]
    lines = set(lines)
    # y축 기준으로 사각 영역을 그룹화한다.
    for line in lines:
        if len(y) == 0:
            y[line] = line
        else:
            matched = False
            for v in y.values():
                if abs(line - v) < 15:
                    y[line] = v
                    matched = True
                    break
            if not matched:
                y[line] = line

    ids = [y[r[1] + r[3]] for r in rects]
    cluster_line = dict()
    for r, i in zip(rects, ids):
        if i not in cluster_line:
            cluster_line[i] = [r]
        else:
            cluster_line[i] += [r]

    cluster = []
    for c in cluster_line.values():
        cluster.append(cluster_x(c))

    min_areas = []
    for g_y in cluster:
        for g_x in g_y:
            x, y, w, h = min_area_rect(g_x)
            size = len(g_x)
            ratio = [x[2] / x[3] for x in g_x]
            avg = np.average(ratio)
            min_areas.append((x, y, w, h, size, avg))

    min_areas = sorted(min_areas, key=lambda t: t[1], reverse=True)
    # y 축이 인접한 것이 있으면 하나로 묶는다.
    multi_rect = cluster_y(min_areas)

    # 자막 영역 후보 찾기
    # 아래 조건에 해당할수록 높은 점수를 부여한다.
    # 화면 하단/가로-세로비/단어수(영역 수)/인접한 y
    if len(multi_rect) > 0:
        rects = np.array(multi_rect)
        w = np.average(rects[:, 2])
        h = np.average(rects[:, 3])
        size = np.average(rects[:, 4])
        ratio = np.average(rects[:, 5])
        y = np.average(rects[:, 1])
        multi_score = w * h * size * ratio * np.sqrt(y) * len(multi_rect)
    else:
        multi_score = 0

    single_score = 0
    single_rect = []
    for area in min_areas:
        score = area[2] * area[3] * area[4] * area[5] * np.sqrt(area[1])
        if score > single_score:
            single_score = score
            single_rect = [area]

    # line 중에서 텍스트 영역이 가능성이 높은 line을 선택한다.
    if len(multi_rect) == 0:
        best_rect = single_rect
    else:
        if multi_score > single_score:
            best_rect = multi_rect
        else:
            best_rect = single_rect

    return cluster, ids, best_rect


def captch_ex(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = remove_noise_and_smooth(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([130, 20, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(img, img, mask=hsv_mask)
    hsv_binary = remove_noise_and_smooth(res)

    contours, dilated = contour_plot_on_text_in_image(hsv_binary)
    rects = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        wh_ratio = w / h

        # Don't plot small false positives that aren't text
        if (w < 10) or (h < 10) or (h > 80) or (wh_ratio < 0.5):
            continue

        rects.append([x, y, w, h])
        # draw rectangle around contour on original image
        # cv2.rectangle(hsv, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cluster, ids, best_rect = cluster_contour(rects)
    for g_y in cluster:
        for g_x in g_y:
            x, y, w, h = min_area_rect(g_x)
            cv2.rectangle(hsv, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if len(best_rect) > 0:
        x, y, w, h = min_area_rect(best_rect)
        cv2.rectangle(hsv, (x, y), (x + w, y + h), (255, 0, 0), 2)

    contours, dilated = contour_plot_on_text_in_image(binary)
    rects = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        wh_ratio = w / h

        # Don't plot small false positives that aren't text
        if (w < 10) or (h < 10) or (h > 80) or (wh_ratio < 0.5):
            continue

        # contour에 0이 아닌 값이 일정 비율 이하인 것은 버린다.
        r = cv2.countNonZero(binary[y:y+h, x:x+w]) / (w * h)
        # if r < 0.15:
        #     continue

        rects.append([x, y, w, h])
        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cluster, ids, best_rect = cluster_contour(rects)
    for g_y in cluster:
        for g_x in g_y:
            x, y, w, h = min_area_rect(g_x)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if len(best_rect) > 0:
        x, y, w, h = min_area_rect(best_rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    hist = cv2.calcHist([gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    images = [res, hsv_binary, hsv, img]
    titles = ['res', 'hsv_binary', 'hsv', 'result']
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i]), plt.title(titles[i]), plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.plot(hist), plt.title('Histogram')
    # plt.imshow(hsv)
    plt.show()
    return img


def find_text_region(img, show_edge=True):
    binary = remove_noise_and_smooth(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([130, 20, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(img, img, mask=hsv_mask)
    hsv_binary = remove_noise_and_smooth(res)
    contours, dilated = contour_plot_on_text_in_image(hsv_binary)

    rects = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        wh_ratio = w / h

        # Don't plot small false positives that aren't text
        if (w < 10) or (h < 10) or (h > 80) or (wh_ratio < 0.5):
            continue

        # contour에 0이 아닌 값이 일정 비율 이하인 것은 버린다.
        # r = cv2.countNonZero(binary[y:y+h, x:x+w]) / (w * h)
        # if r < 0.15:
        #     continue

        rects.append([x, y, w, h])

    cluster, ids, best_rect = cluster_contour(rects)
    if len(best_rect) > 0:
        x, y, w, h = min_area_rect(best_rect)
        text_region = [x, y, x + w, y + h]
        if show_edge:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    else:
        text_region = []

    return text_region, img


def image_preprocess(img, morpholog=False):
    height, width, _ = img.shape
    # img = ocr.adjust_image(img)
    img = remove_noise_and_smooth(img, morpholog)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


if __name__ == "__main__":
    file_name = '../capture/game4.jpg'
    img = cv2.imread(file_name)
    captch_ex(img)
    # rg, img = find_text_region(img)
    # plt.imshow(img)
    # plt.show()
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

