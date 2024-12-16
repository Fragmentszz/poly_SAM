import os
import shutil
import numpy as np
import cv2
def decline_rate(image,x_rate1,x_rate2,y_rate1,y_rate2):
    pos = np.zeros_like(image).astype(np.uint8)
    max_x = np.where(image > 128)[0].max()
    min_x = np.where(image > 128)[0].min()
    max_y = np.where(image > 128)[1].max()
    min_y = np.where(image > 128)[1].min()
    rand_percent_x = np.random.random() * (x_rate2 - x_rate1) + x_rate1
    rand_percent_y = np.random.random() * (y_rate2 - y_rate1) + y_rate1
    

    # �������ѡ�������
    selected_x = int((max_x - min_x) * rand_percent_x)
    selected_y = int((max_y - min_y) * rand_percent_y)


    start_x = np.random.randint(min_x, max_x - selected_x - 1)
    start_y = np.random.randint(min_y, max_y - selected_y - 1)
    # ����pos����
    pos[start_x:selected_x+start_x,start_y:start_y+selected_y] = 255

    selected_image = cv2.bitwise_and(image, pos)
    return selected_image


def trans_gt_to_mask(gt):
    gt = gt.copy()
    gt.astype(np.uint8)
    background = np.where(gt == 0, 255, 0).astype(np.uint8)
    # Ѱ�����������ķǷָ�����
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background, connectivity=4)
    if num_labels > 1:
        # ѡ��һ���������磬ѡ��������ķǷָ�����
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    selected_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    selected_mask = decline_rate(selected_mask,0.25,0.75,0.25,0.75)
    # gt = decline_rate(gt,0.25,0.75,0.25,0.75)
    mask = np.bitwise_or(gt, selected_mask)
    return mask




def merge_dataset(src_dir,dst_dir):
    # image_path = os.path.join(Test_dir,'images')
    # mask_path = os.path.join(Test_dir,'masks')
    
    img_dir = ['images','masks','TestDataset']
    dst_list = ['images','gts','masks','grays']
    for img in dst_list:
        if not os.path.exists(os.path.join(dst_dir,img)):
            os.mkdir(os.path.join(dst_dir,img))
    for sub_name in os.listdir(src_dir):
        # if sub_name in img_dir:
        #     continue
        sub_dir = os.path.join(src_dir, sub_name)
        print(sub_dir)
        if not os.path.isdir(sub_dir):
            continue
        cnt = 0
        sub_images = os.path.join(sub_dir, 'images')
        sub_masks = os.path.join(sub_dir, 'masks')
        print(sub_images,sub_masks)
        for img in os.listdir(sub_images):
            if img.endswith('.png') or img.endswith('.jpg'):
                mask_path = os.path.join(sub_masks, img)
                img_path = os.path.join(sub_images, img)
                image = cv2.imread(img_path)
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = trans_gt_to_mask(gt)
                img_name = sub_name + '_' + str(cnt) + '.png'
                im_li = [image, gt, mask, gray]
                for i, im in enumerate(im_li):
                    cv2.imwrite(os.path.join(dst_dir, dst_list[i], img_name), im)
                cnt += 1
        

def merge_train_data(src_dir,dst_dir):
    img_dir = ['images','masks']
    dst_list = ['images','gts','masks','grays']
    for img in dst_list:
        if not os.path.exists(os.path.join(dst_dir,img)):
            os.mkdir(os.path.join(dst_dir,img))
    cnt = 0
    sub_images = os.path.join(src_dir, 'images')
    sub_masks = os.path.join(src_dir, 'masks')
    for img in os.listdir(sub_images):
        if img.endswith('.png') or img.endswith('.jpg'):
            mask_path = os.path.join(sub_masks, img)
            img_path = os.path.join(sub_images, img)
            image = cv2.imread(img_path)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = trans_gt_to_mask(gt)
            img_name = str(cnt) + '.png'
            im_li = [image, gt, mask, gray]
            for i, im in enumerate(im_li):
                cv2.imwrite(os.path.join(dst_dir, dst_list[i], img_name), im)
            cnt += 1



def merge_test_data_fl(src_dir,dst_dir):
    img_dir = ['images','masks','TestDataset']
    dst_list = ['images','gts','masks','grays']
    
    for sub_name in os.listdir(src_dir):
        sub_dir = os.path.join(src_dir, sub_name)
        if not os.path.isdir(sub_dir):
            continue
        cnt = 0
        sub_images = os.path.join(sub_dir, 'images')
        sub_masks = os.path.join(sub_dir, 'masks')
        dst_path = os.path.join(dst_dir,sub_name)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for img in dst_list:
            if not os.path.exists(os.path.join(dst_path,img)):
                os.mkdir(os.path.join(dst_path,img))
                print(os.path.join(dst_path,img))
        print(dst_path)
        for img in os.listdir(sub_images):
            if img.endswith('.png') or img.endswith('.jpg'):
                mask_path = os.path.join(sub_masks, img)
                img_path = os.path.join(sub_images, img)
                image = cv2.imread(img_path)
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = trans_gt_to_mask(gt)
                img_name = str(cnt) + '.png'
                im_li = [image, gt, mask, gray]
                for i, im in enumerate(im_li):
                    # print(os.path.join(dst_path, dst_list[i], img_name))
                    cv2.imwrite(os.path.join(dst_path, dst_list[i], img_name), im)
                cnt += 1

if __name__ == '__main__':
    # merge_dataset('/root/autodl-tmp/dataset/TestDataset/TestDataset','/root/autodl-tmp/dataset/polyp/test')
    # merge_train_data('/root/autodl-tmp/dataset/TrainDataset','/root/autodl-tmp/dataset/polyp/train')
    merge_test_data_fl('/root/autodl-tmp/dataset/TestDataset/TestDataset','/root/autodl-tmp/dataset/polyp/test')