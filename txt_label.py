import glob
import os
from tqdm import tqdm


def work1():
    root = "/workspace/data/cls_1207_data/5-FourBall-Continuous-20-03-15_Single-1"

    v_set = set()

    for tag in tqdm(os.listdir(root)):
        input_file_path = glob.glob(f'{root}/{tag}/*_data.txt')[0]
        output_file_path = f'{root}/{tag}/output_0.5.txt'

        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            frame_counter = 0
            for line in infile:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                _, value = parts
                frame_label = f"frame_{frame_counter}"
                # value_rounded = round(float(value))
                value = float(value)
                int_value = int(value)
                diff = value - int_value
                if diff < 0.25:
                    value_rounded = int_value
                elif diff < 0.75:
                    value_rounded = int_value + 0.5
                else:
                    value_rounded = int_value + 1
                print(value, value_rounded)
                v_set.add(value_rounded)

                outfile.write(f"{frame_label} {value_rounded}\n")
                frame_counter += 1

    print(v_set, len(v_set))


def work2():
    root = "/workspace/data/cls_1207_data/5-FourBall-Continuous-20-03-15_Single-1"

    for tag in tqdm(os.listdir(root)):
        input_file_path = glob.glob(f'{root}/{tag}/*_data.txt')[0]
        output_file_path = f'{root}/{tag}/output_float.txt'

        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            frame_counter = 0
            for line in infile:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                _, value = parts
                frame_label = f"frame_{frame_counter}"
                # value_rounded = round(float(value))
                value = float(value)
                outfile.write(f"{frame_label} {value}\n")
                frame_counter += 1


if __name__ == "__main__":
    work2()