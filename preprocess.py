import glob
import shutil

import cv2
import os
import numpy as np
from tqdm import tqdm

root = "E:/Desktop/Individual Project/Data/5-FourBall-Continuous-20-03-15_Single-1"
# root = "/workspace/data/cls_1207_data/5-FourBall-Continuous-20-03-15_Single-1"
tag_list = sorted(os.listdir(root))

for tag in tag_list:
    video_path = glob.glob(os.path.join(root, tag, '*.avi'))[0]

    # Output folder paths
    saved_root = os.path.dirname(video_path)
    image_folder_path = os.path.join(saved_root, 'output_images')
    contours_folder_path = os.path.join(saved_root, 'contours_centroid')
    crop_folder_path = os.path.join(saved_root, 'cropped_images')
    annotated_folder_path = os.path.join(saved_root, 'annotated_images')

    output_text_file = os.path.join(saved_root, 'contours_centroid/centroid_coordinates.txt')

    # Parameters for the Gaussian filter
    gaussian_kernel_size = (15, 15)
    gaussian_sigma = 0.5

    # Threshold and offset for binary segmentation
    offset_value = 8

    # Cropping size and bounding box
    crop_size = 230
    bounding_box_margin = 10

    # Read the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video. {tag}")
        continue

    for d in [image_folder_path, contours_folder_path, crop_folder_path, annotated_folder_path]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    with open(output_text_file, 'w') as file:
        file.write("Frame\tX_centroid\tY_centroid\n")


        for frame_count in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # print(f"Processing Frame {frame_count}")
            output_image_path = os.path.join(image_folder_path, f'frame_{frame_count}.png')
            cv2.imwrite(output_image_path, frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian_frame = cv2.GaussianBlur(gray_frame, gaussian_kernel_size, gaussian_sigma)

            sobel_x = cv2.Sobel(gaussian_frame, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gaussian_frame, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
            edges = np.uint8(gradient_magnitude)

            # Apply binary segmentation (Otsu's + offset)

            _, binary_frame = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # binary_frame = cv2.threshold(gaussian_frame, _ + offset_value, 255, cv2.THRESH_BINARY_INV)[1]
            # binary_frame = cv2.threshold(gaussian_frame, 110, 255, cv2.THRESH_BINARY_INV)[1]

            # Morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)

            # Find contours
            cnts, _ = cv2.findContours(closed_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                print(f"No contours found in Frame {frame_count}")
                continue

            # Get the largest contour
            sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            c = sorted_cnts[0]
            if c[0, 0, 0] == 0 and c[0, 0, 1] == 0:
                c = sorted_cnts[1]

            # Create a mask for the contour
            mask = np.zeros_like(gaussian_frame)
            cv2.drawContours(mask, [c], -1, 255, -1)

            # Intensity-weighted centroid
            C = gaussian_frame * (mask // 255)
            C_sum = np.sum(C)

            x_indices, y_indices = np.indices(gaussian_frame.shape)
            x_c = np.sum(C * x_indices) / C_sum
            y_c = np.sum(C * y_indices) / C_sum

            file.write(f'{frame_count} {y_c / 5:.6f} {x_c / 5:.6f}\n')

            # Draw the contours and centroid on the image
            image = cv2.cvtColor(gaussian_frame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (int(y_c), int(x_c)), 5, (0, 0, 255), -1)
            cv2.putText(image, "Centroid", (int(y_c) - 25, int(x_c) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            output_image_path = os.path.join(contours_folder_path, f'frame_{frame_count}.png')
            cv2.imwrite(output_image_path, image)

            # Crop the image
            half_crop_size = crop_size // 2

            x_start = int(max(x_c - half_crop_size, 0))
            x_end = int(min(x_c + half_crop_size, image.shape[0]))
            y_start = int(max(y_c - half_crop_size, 0))
            y_end = int(min(y_c + half_crop_size, image.shape[1]))

            cropped_image = gaussian_frame[x_start:x_end, y_start:y_end]

            cropped_image_path = os.path.join(crop_folder_path, f'{frame_count}.png')
            cv2.imwrite(cropped_image_path, cropped_image)

            # Annotate original frame with bounding box and tag
            annotated_image = frame.copy()

            expanded_x_start = max(x_start - bounding_box_margin, 0)
            expanded_x_end = min(x_end + bounding_box_margin, frame.shape[0])
            expanded_y_start = max(y_start - bounding_box_margin, 0)
            expanded_y_end = min(y_end + bounding_box_margin, frame.shape[1])

            cv2.rectangle(annotated_image, (expanded_y_start, expanded_x_start), (expanded_y_end, expanded_x_end), (255, 0, 0), 2)

            tag_height = 25
            tag_width = 110
            tag_color = (255, 0, 0)
            tag_start = (expanded_y_start, expanded_x_start + tag_height)

            if tag_start[1] < 0:
                tag_start = (expanded_y_start, expanded_x_end + tag_height)

            # Draw the tag rectangle just above the bounding box
            cv2.rectangle(annotated_image, tag_start, (tag_start[0] + tag_width, tag_start[1] - tag_height), tag_color, -1)

            cv2.putText(annotated_image, "Microrobot", (tag_start[0] + 5, tag_start[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(annotated_image, (int(y_c), int(x_c)), 5, (255, 0, 0), -1)
            cv2.putText(annotated_image, "Centroid", (int(y_c) - 25, int(x_c) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            annotated_image_path = os.path.join(annotated_folder_path, f'annotated_{frame_count}.png')
            cv2.imwrite(annotated_image_path, annotated_image)

            # frame_count += 1

    cap.release()
cv2.destroyAllWindows()
