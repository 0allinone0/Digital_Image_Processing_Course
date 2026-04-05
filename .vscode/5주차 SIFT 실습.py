from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel

#이미지 불러오기
img1 = Image.open("Lenna.png").convert("L")   #convert?
#Image 변경하기
img2 = img1.rotate(5).crop((30, 30, img1.width - 30, img1.height - 30))

#img1.show()
#img2.show()

#hessian filter
def hessian_filter(dog, x, y, edge_threshold=10):
    Dxx = dog[y, x+1] + dog[y, x-1] - 2 * dog[y, x]
    Dyy = dog[y+1, x] + dog[y-1, x] - 2 * dog[y, x]
    Dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) / 4.0

    Tr_H = Dxx + Dyy
    Det_H = Dxx * Dyy - Dxy**2

    R = (Tr_H**2) / Det_H
    r = edge_threshold
    R_thresh = ((r + 1)**2) / r

    return R <R_thresh

#get descriptor 함수
def get_descriptor(img, x, y, size=8):
    patch = img[y-size//2:y+size//2, x-size//2:x+size//2]
    if patch.shape != (size, size):
        return None

    gx = sobel(patch, axis=1)   #sobel?
    gy = sobel(patch, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 360

    hist, bin_edges = np.histogram(angle, bins=36, range=(0, 360), weights=mag)
    dominant_angle = bin_edges[np.argmax(hist)]  #가장 dominant한 angle찾기
    aligned_angle = (angle - dominant_angle + 360) % 360   #normalization
    hist, _ = np.histogram(aligned_angle, bins=8, range=(0, 360), weights=mag)   #  _ 이게 뭐지?

    vector = hist / (np.linalg.norm(hist) + 1e-6)
    vector = np.clip(vector, 0, 0.2)
    vector = vector / (np.linalg.norm(vector) + 1e-6)

    return vector

#Gussian filter (DoG), Descriptor
#img1

def find_descriptor(img1):
    img1_np = np.array(img1, dtype=np.float32) / 255.0
    blur1_img1 = gaussian_filter(img1_np, sigma=1.0)
    blur2_img1 = gaussian_filter(img1_np, sigma=2.0)
    dog1 = blur1_img1 - blur2_img1

    keypoints = []
    threshold = 0.03
    for y in range(1, dog1.shape[0]-1):
        for x in range(1, dog1.shape[1]-1):
            patch = dog1[y-1:y+2, x-1:x+2]
            center = dog1[y, x]
            if np.abs(center) > threshold and (center == patch.max() or center == patch.min()):
                keypoints.append((x,y))

    #final keypoints 찾기
    final_keypoints = []
    for x, y in keypoints:
        if 1 < x <dog1.shape[1] -1 and 1 < y < dog1.shape[0] - 1:
            if hessian_filter(dog1, x, y):
                final_keypoints.append((x,y))

    #descriptors 찾기
    descriptors = []
    filtered_kps = []
    for x, y in final_keypoints:
        desc = get_descriptor(img1_np, x, y)
        if desc is not None:
            descriptors.append(desc)
            filtered_kps.append((x, y))

    return filtered_kps, descriptors


def match_images(desc1, desc2):
    matches = []
    threshold = 0.001
    
    for i, f1 in enumerate(desc1):
        diff = np.linalg.norm(desc2 - f1, axis=1)
        sorted_index = np.argsort(diff)
        best = sorted_index[0]
        second_best = sorted_index[1]

        ratio = diff[best] /(diff[second_best] + 1e-6)

        if ratio < threshold:
            f2 = desc2[best]
            rev_diff = np.linalg.norm(desc1 - f2, axis=1)
            rev_sorted_index = np.argsort(rev_diff)
            rev_best = rev_sorted_index[0]
            rev_second_best = rev_sorted_index[1]

            rev_ratio = rev_diff[rev_best] / (rev_diff[rev_second_best] + 1e-6)

            if rev_best == i and rev_ratio < threshold:
                matches.append((i, best))

    return matches


#point 및 특징
pt1, desc1 = find_descriptor(img1)
pt2, desc2 = find_descriptor(img2)

matches = match_images(desc1, desc2)




#visualization
from PIL import ImageDraw
h, w = img1.size[1], img1.size[0]
h2, w2 = img2.size[1], img2.size[0]

out_img = Image.new('L', (w+w2, max(h, h2)))
out_img.paste(img1, (0, 0))
out_img.paste(img2, (w,0))

draw = ImageDraw.Draw(out_img)
point_color = 'cyan'
line_color = 'red'
for i, j in matches:
    x1, y1 = pt1[i]
    x2, y2 = pt2[j]

    r = 2
    draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill = point_color)
    draw.ellipse((x2 + w - r, y2 - r, x2 + w +r, y2 + r), fill = point_color)
    draw.line((x1, y1, x2 + w, y2), fill=line_color, width=1)

out_img.show()




