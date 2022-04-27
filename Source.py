import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.ndimage.morphology import binary_closing
from scipy.spatial import distance
import time

def extract_contours():
    pic_objects = ['circle.jpg', 'triangle.jpg', 'coin.jpg', 'deer.jpg', 'square.jpg', 'flash.jpg', 'key.jpg', 'star.jpg', 'pencil.jpg']
    pic_objects = [ 'objects\\' + name for name in pic_objects]
    obj_contours = []
    for pic_obj in pic_objects:
        img = cv.imread(pic_obj)
        img_to_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img_to_gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        bin_mask_1 = 0.7 * np.mean(img_to_gr) - img_to_gr > 0
        bin_mask_2 = (cv.inRange(img_to_hsv, (0, 25, 0), (200, 255, 255)) > 0) | (cv.inRange(img_to_hsv, (0, 0, 0), (200, 255, 120)) > 0)
        bin_mask_3 = binary_closing(bin_mask_1 | bin_mask_2, np.ones((25, 25)))
        contours, hierarchy = cv.findContours(255 * np.uint8(bin_mask_3), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #reference pics' have this hierarchy:boundary->sheet->object, meaning that the third contour is the one to choose
        obj_contours.append(contours[2])
    return obj_contours

def find_poly_cnt(contours, hierarchy):
    pic_cy = 2016 # 4032 / 2 pic_barycenter y-coord
    for i in range(len(contours)):
        if hierarchy[0, i, -1] == 0:   #check if contour's parent is outermost pic contour 
            M = cv.moments(contours[i])
            cy = int(M['m01'] / M['m00'])
            if cy < pic_cy:   #check if contour represents UPPER sheet
                outer_cnt_ind = hierarchy[0, i, -2]
                inner_cnt_ind = hierarchy[0, outer_cnt_ind, -2]
    return inner_cnt_ind

def find_object_cnts(contours, hierarchy):
    pic_cy = 2016 # 4032 / 2 pic_barycenter y-coord
    obj_cnts = []
    for i in range(len(contours)):
        if hierarchy[0, i, -1] == 0:   #check if contour's parent is outermost pic contour 
            M = cv.moments(contours[i])
            cy = int(M['m01'] / M['m00'])
            if cy > pic_cy:
                noisy_cnt_area_threshold = 0.01 * cv.contourArea(contours[i])
                j = hierarchy[0, i, -2]
                while j != -1:
                    if cv.contourArea(contours[j]) > noisy_cnt_area_threshold:
                        obj_cnts.append(j)
                    j = hierarchy[0, j, 0]
    return obj_cnts


class Objects_analyzer:

    def __init__(self, pic_contours, pic_cnt_hierarchy, pic_obj_cnts, reference_obj_list):
        self.circ_cnt, self.tri_cnt, self.coin_cnt, self.deer_cnt, self.sq_cnt, self.fl_cnt, self.key_cnt, self.star_cnt, self.pencil_cnt = reference_obj_list
        self.pic_contours = pic_contours
        self.pic_cnt_hierarchy = pic_cnt_hierarchy
        self.pic_obj_cnts = set(pic_obj_cnts)
        self.pic_recognized_objs = None
        
    def check_for_key(self):
        if len(self.pic_obj_cnts) == 0:
            return -1
        for i in self.pic_obj_cnts:
            if self.pic_cnt_hierarchy[0, i, -2] != -1:
                return i
        return -1
    
    def check_for_triangle(self):
        if len(self.pic_obj_cnts) == 0:
            return -1
        for i in self.pic_obj_cnts:
            obj_area = cv.contourArea(self.pic_contours[i])
            enc_trian_area = cv.minEnclosingTriangle(self.pic_contours[i])[0]
            if np.abs(obj_area- enc_trian_area) / max(enc_trian_area, obj_area) <= 0.125:
                return i
        return -1
    
    def check_for_coin_and_circle(self):   #   return coin_ind, circle_ind
        if len(self.pic_obj_cnts) == 0:
            return -1, -1
        round_objs = []
        for i in self.pic_obj_cnts:
            obj_area = cv.contourArea(self.pic_contours[i])
            enc_round_area = cv.minEnclosingCircle(self.pic_contours[i])[1] ** 2 * np.math.pi
            if np.abs(obj_area- enc_round_area) / max(enc_round_area, obj_area) <= 0.125:
                round_objs.append(i)
        round_obj_count = len(round_objs)
        if round_obj_count == 2:
            area_1 = cv.contourArea(self.pic_contours[round_objs[0]])
            area_2 = cv.contourArea(self.pic_contours[round_objs[1]])
            if area_1 < area_2:
                return round_objs[0], round_objs[1]
            return round_objs[1], round_objs[0]
        if round_obj_count == 1:
            ref_circ_diff = np.abs(cv.contourArea(self.circ_cnt) - cv.contourArea(self.pic_contours[round_objs[0]]))
            ref_coin_diff = np.abs(cv.contourArea(self.coin_cnt) - cv.contourArea(self.pic_contours[round_objs[0]]))
            if ref_circ_diff > ref_coin_diff:
                return round_objs[0], -1
            return -1, round_objs[0]
        return -1, -1
    
    def check_for_deer_and_star(self):
        if len(self.pic_obj_cnts) == 0:
            return -1, -1                #   check for two easily distinguishable non-convex objects (assuming that the
        strongly_non_convex_objs = []    #   key-object recognition procedure was undertaken)
        for i in self.pic_obj_cnts:      #   return deer_ind, star_ind
            obj_area = cv.contourArea(self.pic_contours[i])
            conv_hull_area = cv.contourArea(cv.convexHull(self.pic_contours[i]))
            if (conv_hull_area - obj_area) / conv_hull_area >= 0.175:
                strongly_non_convex_objs.append(i)
        str_non_conv_obj_count = len(strongly_non_convex_objs)
        if str_non_conv_obj_count == 1:
            deer_dissimilarity = cv.matchShapes(self.deer_cnt, self.pic_contours[strongly_non_convex_objs[0]], 1,0)
            star_dissimilarity = cv.matchShapes(self.star_cnt, self.pic_contours[strongly_non_convex_objs[0]], 1,0)
            if deer_dissimilarity < star_dissimilarity:
                return strongly_non_convex_objs[0], -1
            return -1, strongly_non_convex_objs[0]
        if str_non_conv_obj_count == 2:
            deer_dissimilarity_obj_1 = cv.matchShapes(self.deer_cnt, self.pic_contours[strongly_non_convex_objs[0]], 1,0)
            deer_dissimilarity_obj_2 = cv.matchShapes(self.deer_cnt, self.pic_contours[strongly_non_convex_objs[1]], 1,0)
            if deer_dissimilarity_obj_1 < deer_dissimilarity_obj_2:
                return strongly_non_convex_objs[0], strongly_non_convex_objs[1]
            return strongly_non_convex_objs[1], strongly_non_convex_objs[0]
        return -1, -1
        
    def check_for_square(self):
        if len(self.pic_obj_cnts) == 0:
            return -1
        for i in self.pic_obj_cnts:
            rect = cv.minAreaRect(self.pic_contours[i])
            box = cv.boxPoints(rect)
            box = np.int0(box)
            side_1_len = np.linalg.norm(box[0] - box[1])
            side_2_len = np.linalg.norm(box[2] - box[1])
            if np.abs(side_2_len - side_1_len) / max(side_1_len, side_2_len) < 0.075:
                return i
        return -1
    
    def check_for_flash_and_pencil(self):
        if len(self.pic_obj_cnts) == 0:
            return -1, -1                       #   return flash_ind, pencil_ind
        rect = cv.minAreaRect(self.pencil_cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        side_1_len = np.linalg.norm(box[0] - box[1])
        side_2_len = np.linalg.norm(box[2] - box[1])
        pencil_side_ratio = max(side_1_len, side_2_len) / min(side_1_len, side_2_len)
        
        cnt_ind = self.pic_obj_cnts.pop()
        rect = cv.minAreaRect(self.pic_contours[cnt_ind])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        side_1_len = np.linalg.norm(box[0] - box[1])
        side_2_len = np.linalg.norm(box[2] - box[1])
        obj_1_side_ratio = max(side_1_len, side_2_len) / min(side_1_len, side_2_len)
        
        if len(self.pic_obj_cnts) == 0:
            rect = cv.minAreaRect(self.fl_cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            side_1_len = np.linalg.norm(box[0] - box[1])
            side_2_len = np.linalg.norm(box[2] - box[1])
            flash_side_ratio = max(side_1_len, side_2_len) / min(side_1_len, side_2_len)
            if np.abs(obj_1_side_ratio - flash_side_ratio) < np.abs(obj_1_side_ratio - pencil_side_ratio):
                return cnt_ind, -1
            return -1, cnt_ind
        
        last_cnt_ind = self.pic_obj_cnts.pop()
        rect = cv.minAreaRect(self.pic_contours[last_cnt_ind])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        side_1_len = np.linalg.norm(box[0] - box[1])
        side_2_len = np.linalg.norm(box[2] - box[1])
        obj_2_side_ratio = max(side_1_len, side_2_len) / min(side_1_len, side_2_len)
        if np.abs(obj_1_side_ratio - pencil_side_ratio) < np.abs(obj_2_side_ratio - pencil_side_ratio):
            return last_cnt_ind, cnt_ind
        return cnt_ind, last_cnt_ind
    
    def analyze_contours(self):
        ans = {}
        cnt_key = self.check_for_key()
        if cnt_key != -1:
            ans['key'] = cnt_key
            self.pic_obj_cnts.remove(cnt_key)
        cnt_tri = self.check_for_triangle()
        if cnt_tri != -1:
            ans['triangle'] = cnt_tri
            self.pic_obj_cnts.remove(cnt_tri)
        cnt_coin, cnt_circ = self.check_for_coin_and_circle()
        if cnt_coin != -1:
            ans['coin'] = cnt_coin
            self.pic_obj_cnts.remove(cnt_coin)
        if cnt_circ != -1:
            ans['circle'] = cnt_circ
            self.pic_obj_cnts.remove(cnt_circ)
        cnt_deer, cnt_star = self.check_for_deer_and_star()
        if cnt_deer != -1:
            ans['deer'] = cnt_deer
            self.pic_obj_cnts.remove(cnt_deer)
        if cnt_star != -1:
            ans['star'] = cnt_star
            self.pic_obj_cnts.remove(cnt_star)
        #print(self.pic_obj_cnts)
        cnt_square = self.check_for_square()
        if cnt_square != -1:
            ans['square'] = cnt_square
            self.pic_obj_cnts.remove(cnt_square)
        cnt_fl, cnt_pencil = self.check_for_flash_and_pencil()
        if cnt_fl != -1:
            ans['flash'] = cnt_fl
        if cnt_pencil != -1:
            ans['pencil'] = cnt_pencil
        self.pic_recognized_objs = ans
        return ans
    
    def necessary_cond_1(self, poly):   #area inequality
        joint_area = 0
        for val in self.pic_recognized_objs.values():
            joint_area += cv.contourArea(self.pic_contours[val])
        return cv.contourArea(poly) > joint_area
    
    def necessary_cond_2(self, poly):   #diam condition
        poly_diam = distance.cdist(poly[:,0], poly[:,0], 'euclidean').max()
        for val in self.pic_recognized_objs.values():
            cur_cnt = self.pic_contours[val]
            max_diam = distance.cdist(cur_cnt[:,0], cur_cnt[:,0], 'euclidean').max()
            if max_diam > poly_diam:
                return False
        return True
    
    
def analyze_pic(img_name, ref_cnts, dpi_val=250):
    img = cv.imread(img_name)
    img_to_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_to_gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bin_mask_1 = 0.7 * np.mean(img_to_gr) - img_to_gr > 0
    bin_mask_2 = (cv.inRange(img_to_hsv, (0, 25, 0), (200, 255, 255)) > 0) | (cv.inRange(img_to_hsv, (0, 0, 0), (200, 255, 120)) > 0)
    bin_mask_3 = binary_closing(bin_mask_1 | bin_mask_2, np.ones((25, 25)))
    contours, hierarchy = cv.findContours(255 * np.uint8(bin_mask_3), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cpy = np.copy(img)
    cpy = cv.cvtColor(cpy, cv.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2, dpi=dpi_val)
    ax[0].imshow(cpy)
    ax[0].axis('off')
    can_be_placed = True if img_name.split('_')[1][0] == 't' else False
    ax[0].set_title('Can be placed: ' + str(can_be_placed))
    poly_contour = find_poly_cnt(contours, hierarchy)
    obj_contours = find_object_cnts(contours, hierarchy)
    obj_analyzer = Objects_analyzer(contours, hierarchy, obj_contours, ref_cnts)
    obj_contours_analyzed = obj_analyzer.analyze_contours()
    for el in obj_contours_analyzed:
        rect = cv.boundingRect(contours[obj_contours_analyzed[el]])
        x,y,w,h = rect
        M = cv.moments(contours[obj_contours_analyzed[el]])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #x+w,y+h
        cv.rectangle(cpy,(x,y),(x+w,y+h),(0,0,255),6)
        cv.putText(cpy,el,(cx, cy),2,4,(255,0,255), 3)
    cv.drawContours( cpy, [contours[poly_contour]], -1, (255,0,0), 4, cv.LINE_AA)
    if not obj_analyzer.necessary_cond_1(contours[poly_contour]):
        ax[1].imshow(cpy)
        ax[1].set_title('Prediction: False\n Area condition failed')
        ax[1].axis('off')
        return can_be_placed, False
    if not obj_analyzer.necessary_cond_2(contours[poly_contour]):
        ax[1].imshow(cpy)
        ax[1].set_title('Prediction: False\n Diameter condition failed')
        ax[1].axis('off')
        return can_be_placed, False
    obj_contours_lines = [contours[val] for val in obj_contours_analyzed.values()]
    obj_contours_lines.sort(reverse=True, key=cv.contourArea)
    res = bottom_left_packing(cpy, obj_contours_lines,  contours[poly_contour])
    prediction = True if res != None else False
    cv.drawContours( cpy, res, -1, (255,0,0), 3, cv.LINE_AA)
    ax[1].imshow(cpy)
    ax[1].set_title('Prediction: ' + str(prediction))
    #plt.subplots_adjust(wspace=0.3)
    ax[1].axis('off')
    return can_be_placed, prediction


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    if angle % 360 == 0:
        return cnt
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def bottom_left_packing(img, obj_contours,  polygon):
    np.random.seed(int(time.time()))
    n_rotations = 4
    result = []
    for i in range(len(obj_contours)):
        M = cv.moments(obj_contours[i])
        center = np.asarray([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.int0)
        obj_contours[i] -= center
    steps_num = 20
    rc = cv.boundingRect(polygon) # x,y,w,h
    top_left = np.asarray((rc[0], rc[1]))
    poly_mask = np.zeros(img.shape[:-1], dtype=np.uint8)
    cv.drawContours(poly_mask, [polygon], -1, 1, -1, cv.LINE_4)
    poly_count = np.sum(poly_mask)
    obj_mask = np.empty(img.shape[:-1], dtype=np.uint8)
    packing_mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    x_step = np.asarray((rc[2] / steps_num, 0))
    y_step = np.asarray((0, rc[3] / steps_num))

    move = lambda c, dir: np.int0(c + dir)
    
    def check_bounds(contour):
        obj_mask.fill(0)
        cv.drawContours(obj_mask, [contour], -1, 1, -1, cv.LINE_4)
        if np.sum(np.logical_or(obj_mask, poly_mask)) > poly_count:
            return False
        if np.sum(np.logical_and(obj_mask, packing_mask)) > 0:
            return False
        return True

    def sift(contour):
        f_shift = False
        shifted = np.copy(contour)
        for i in range(steps_num):
            if np.max(shifted[..., 1]) > rc[1] + rc[3]:
                break
            shifted = move(shifted, y_step)
            if check_bounds(shifted):
                contour[...] = shifted
                f_shift = True
            elif f_shift:
                break
        shifted = np.copy(contour)
        for i in range(steps_num):
            if np.min(shifted[..., 0]) < rc[0]:
                break
            shifted = move(shifted, -x_step)
            if check_bounds(shifted):
                contour[...] = shifted
                f_shift = True
            elif f_shift:
                break
        return f_shift
    
    for c in obj_contours:
        placed = False
        for i in range(1, steps_num):
            for j in range(n_rotations + 1):
                cur_contour = np.copy(c) + top_left
                if j > 0:
                    cur_contour = rotate_contour(cur_contour, np.random.uniform(0, 360))
                cur_contour = move(cur_contour, i * x_step)
            
                f_shifted = sift(cur_contour)
                while f_shifted:
                    placed = f_shifted or placed
                    f_shifted = sift(cur_contour)
                if placed:
                    cv.drawContours(packing_mask, [cur_contour], -1, 1, -1, cv.LINE_4)
                    result.append(cur_contour)
                    break
            if placed:
                break
        if not placed:
            return None
    
    '''
    for c in obj_contours:
        placed = False
        for i in range(1, steps_num):
            cur_contour = np.copy(c) + top_left
            cur_contour = move(cur_contour, i * x_step)

            f_shifted = sift(cur_contour)
            while f_shifted:
                placed = f_shifted or placed
                f_shifted = sift(cur_contour)
            if placed:
                cv.drawContours(packing_mask, [cur_contour], -1, 1, -1, cv.LINE_4)
                result.append(cur_contour)
                break
        if not placed:
            return None
    '''
    
    return result