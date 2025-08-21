from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2
from rosbags.image import message_to_cvimage # pip install rosbags-image
import os
from tqdm import tqdm
import numpy as np


def decode_depth_msg(msg):
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(';')
    
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if 'compressedDepth' not in compr_type:
        raise Exception("Compression type is not 'compressedDepth'."
                        "You probably subscribed to the wrong topic.")

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth_img_raw is None:
        raise Exception("Could not decode compressed depth image."
                        "You may need to change 'depth_header_size'!")

    if depth_fmt == "16UC1":
       return depth_img_raw
    else:
        raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")


def save_resized(img, width, save_path, frame):
    resized_img = cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{save_path}_%06i.png' % frame, resized_img)


# def threshold_normalize(img, min_val, max_val):
#     return np.where((min_val < img) & (img < max_val), img, min_val)

def main():
    rosbag_path = '/mnt/hdd2/yizhouhu/bags/lego_c39c74af_trial_2_20_2025-08-18-19-23-49.bag' #/home/mfi/repos/ros1_ws/src/philip/data/lego_cliff_2025-06-16-17-13-46.bag'
    image_output_dir = '/home/mfi/repos/ros1_ws/src/mmliu/lego-failure/data/temp/'
    N = 10 # 1 image per N frames
    width = 1000


    IMAGE_TYPE = 'gen3' # 'gen3' | 'sideview_depth' | 'sideview_color'
    
    for ROBOT in ['architect', 'destroyer']:
        save_path = os.path.join(image_output_dir, f'{ROBOT}_{os.path.basename(rosbag_path).split(".")[0]}')
        # extract images
        with AnyReader([Path(rosbag_path)]) as reader:
            frame = 0
            for connection in reader.connections:
                print(connection.topic, connection.msgtype)
            print()
            for connection, timestamp, rawdata in tqdm(reader.messages()):
                # gen3 color images (both cameras)
                if IMAGE_TYPE == 'gen3':
                    if f'yk_{ROBOT}/gen3_image' in connection.topic:
                        if frame%N == 0:
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            img = message_to_cvimage(msg, 'bgr8') # change encoding type if needed
                            resized_img = cv2.resize(img, (1000, int(img.shape[0]*width/img.shape[1])), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(f'{save_path}_%06i.png' % frame, resized_img)
                            print(f'{save_path}_%06i.png' % frame)
                        frame += 1

                # sideview depth image
                if IMAGE_TYPE == 'sideview_depth':
                    if f'cam_{ROBOT}/depth/image_raw/compressedDepth' in connection.topic:
                        if frame%N == 0:
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            img = decode_depth_msg(msg)
                            img = threshold_normalize(img, 700, 1000) ###
                            print('max:', img.max())
                            normalized_img = (img - img.min()) / (img.max()-img.min()) * 255.0
                            cv2.imwrite(f'{save_path}_depth_%06i.png' % frame, normalized_img)
                        frame += 1

                # sideview color image
                if IMAGE_TYPE == 'sideview_color':
                    if f'cam_{ROBOT}/color/image_raw/compressed' in connection.topic:
                        if frame%N == 0:
                            msg = reader.deserialize(rawdata, connection.msgtype)
                            img = message_to_cvimage(msg, 'bgr8') # change encoding type if needed
                            cv2.imwrite(f'{save_path}_%06i.png' % frame, img)
                        frame += 1


if __name__ == '__main__':
    main()
