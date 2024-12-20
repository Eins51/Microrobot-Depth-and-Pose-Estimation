import cv2
import os
import csv

input_root_folder = 'E:/Desktop/Individual Project/Data/5-FourBall-Continuous-20-03-15_Single-1'
output_root_folder = 'E:/Desktop/Individual Project/Data/5-FourBall-Continuous-20-03-15_Single-1/planar_rotation_dataset'
os.makedirs(output_root_folder, exist_ok=True)

# Rotation angles from 0° to 360° in 5° increments
angles = range(0, 360, 5)

for angle in angles:
    folder_name = f"Y{angle}"
    angle_folder = os.path.join(output_root_folder, folder_name)
    os.makedirs(angle_folder, exist_ok=True)

txt_file_path = os.path.join(output_root_folder, "image_rotation_data.txt")
csv_file_path = os.path.join(output_root_folder, "image_rotation_data.csv")

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Open txt and csv files for writing
with open(txt_file_path, 'w') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    for tag_folder in os.listdir(input_root_folder):
        cropped_images_folder = os.path.join(input_root_folder, tag_folder, "cropped_images")
        if not os.path.exists(cropped_images_folder):
            continue

        # Process each cropped image in the folder
        for filename in os.listdir(cropped_images_folder):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                img_path = os.path.join(cropped_images_folder, filename)
                image = cv2.imread(img_path)

                # Save the original image to Y0
                folder_name = "Y0"
                output_filename = f"{os.path.splitext(filename)[0]}_{tag_folder}.png"
                output_path = os.path.join(output_root_folder, folder_name, output_filename)
                cv2.imwrite(output_path, image)
                print(f"Saved original {output_filename} in {folder_name}")

                # Record original data in txt and csv
                txt_file.write(f"{output_filename}\t0\n")
                csv_writer.writerow([output_filename, 0])

                # Apply rotations and save each rotated image
                for angle in angles[1:]:
                    rotated_image = rotate_image(image, angle)

                    folder_name = f"Y{angle}"
                    output_filename = f"{os.path.splitext(filename)[0]}_{tag_folder}.png"
                    output_path = os.path.join(output_root_folder, folder_name, output_filename)

                    # Save rotated image
                    cv2.imwrite(output_path, rotated_image)
                    print(f"Saved {output_filename} in {folder_name}")

                    # Record rotated data in txt and csv
                    txt_file.write(f"{output_filename}\t{angle}\n")
                    csv_writer.writerow([output_filename, angle])

print("All rotations completed, and data saved in both txt and csv files.")
