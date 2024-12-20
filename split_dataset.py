import glob
import os
import random


def create_depth_dataset(depth_range, name="depth"):
    root = '/workspace/data/cls_1207_data/5-FourBall-Continuous-20-03-15_Single-1'
    save_dir = os.path.dirname(root)
    # create pose dataset
    pose_list = sorted(os.listdir(root))

    all_depths = []

    depth_range = [float(x) for x in depth_range]

    for tag in pose_list:
        crop_dir = f"{root}/{tag}/cropped_images"
        assert os.path.isdir(crop_dir)
        depth_file = f"{root}/{tag}/output.txt"
        with open(depth_file, 'r') as f:
            lines = f.readlines()
            f.close()

        for line in lines:
            frame_name, depth = line.strip().split(' ')
            d_cls = depth_range.index(float(depth))
            frame_name = frame_name.replace('frame_', '') + ".png"
            path = os.path.join(crop_dir, frame_name)
            assert os.path.exists(path)

            all_depths.append(f"{tag}/cropped_images/{frame_name} {d_cls}")

    random.shuffle(all_depths)

    # 70-30
    split = int(len(all_depths) * 0.7)

    all_train = all_depths[:split]
    all_val = all_depths[split:]

    print(len(all_train), len(all_val))

    with open(os.path.join(save_dir, f'{name}_train.txt'), 'w+') as f:
        for line in all_train:
            f.write(line + "\n")

    with open(os.path.join(save_dir, f'{name}_val.txt'), 'w+') as f:
        for line in all_val:
            f.write(line + "\n")

    # # 60-20-20
    # split1 = int(len(all_depths) * 0.6)
    # split2 = int(len(all_depths) * 0.8)
    # all_train = all_depths[:split1]
    # all_val = all_depths[split1:split2]
    # all_test = all_depths[split2:]
    #
    # print(len(all_train), len(all_val), len(all_test))
    #
    # with open(os.path.join(save_dir, f'{name}_train.txt'), 'w+') as f:
    #     for line in all_train:
    #         f.write(line + "\n")
    #
    # with open(os.path.join(save_dir, f'{name}_val.txt'), 'w+') as f:
    #     for line in all_val:
    #         f.write(line + "\n")
    #
    # with open(os.path.join(save_dir, f'{name}_test.txt'), 'w+') as f:
    #     for line in all_test:
    #         f.write(line + "\n")


def create_pose_dateset():
    root = '/workspace/data/cls_1207_data/5-FourBall-Continuous-20-03-15_Single-1'
    save_dir = os.path.dirname(root)
    # create pose dataset
    pose_list = sorted(os.listdir(root))

    all_train = []
    all_val = []

    for cls_i, tag in enumerate(pose_list):
        crop_dir = f"{root}/{tag}/cropped_images"
        assert os.path.isdir(crop_dir)
        pose_samples = []
        for file in os.listdir(crop_dir):
            path = f"{tag}/cropped_images/{file}"
            sample = (path, tag, cls_i)
            pose_samples.append(sample)
        random.shuffle(pose_samples)

        # 70-30
        split = int(len(pose_samples) * 0.7)

        all_train.extend(pose_samples[:split])
        all_val.extend(pose_samples[split:])

        # # 60-20-20
        # split1 = int(len(pose_samples) * 0.6)
        # split2 = int(len(pose_samples) * 0.8)
        # all_train.extend(pose_samples[:split1])
        # all_val.extend(pose_samples[split1:split2])
        # all_test.extend(pose_samples[split2:])

    print(len(all_train), len(all_val))

    with open(os.path.join(save_dir, 'pose_train.txt'), 'w+') as f:
        for path, tag, cls_i in all_train:
            f.write(f"{path} {tag} {cls_i}\n")

    with open(os.path.join(save_dir, 'pose_val.txt'), 'w+') as f:
        for path, tag, cls_i in all_val:
            f.write(f"{path} {tag} {cls_i}\n")

    # with open(os.path.join(save_dir, 'pose_test.txt'), 'w+') as f:
    #     for path, tag, cls_i in all_test:
    #         f.write(f"{path} {tag} {cls_i}\n")


def create_float_depth_dataset(name="depth_float"):
    root = '/workspace/data/cls_1207_data/5-FourBall-Continuous-20-03-15_Single-1'
    save_dir = os.path.dirname(root)
    # create pose dataset
    pose_list = sorted(os.listdir(root))
    all_depths = []
    for tag in pose_list:
        crop_dir = f"{root}/{tag}/cropped_images"
        assert os.path.isdir(crop_dir)
        depth_file = f"{root}/{tag}/output_float.txt"
        with open(depth_file, 'r') as f:
            lines = f.readlines()
            f.close()

        for line in lines:
            frame_name, depth = line.strip().split(' ')
            depth = float(depth)
            frame_name = frame_name.replace('frame_', '') + ".png"
            path = os.path.join(crop_dir, frame_name)
            assert os.path.exists(path)

            all_depths.append(f"{tag}/cropped_images/{frame_name} {depth}")

    random.shuffle(all_depths)

    # 70-30
    split = int(len(all_depths) * 0.7)

    all_train = all_depths[:split]
    all_val = all_depths[split:]

    print(len(all_train), len(all_val))

    with open(os.path.join(save_dir, f'{name}_train.txt'), 'w+') as f:
        for line in all_train:
            f.write(line + "\n")

    with open(os.path.join(save_dir, f'{name}_val.txt'), 'w+') as f:
        for line in all_val:
            f.write(line + "\n")

    # # 60-20-20
    # split1 = int(len(all_depths) * 0.6)
    # split2 = int(len(all_depths) * 0.8)
    # all_train = all_depths[:split1]
    # all_val = all_depths[split1:split2]
    # all_test = all_depths[split2:]
    #
    # print(len(all_train), len(all_val), len(all_test))
    #
    # with open(os.path.join(save_dir, f'{name}_train.txt'), 'w+') as f:
    #     for line in all_train:
    #         f.write(line + "\n")
    #
    # with open(os.path.join(save_dir, f'{name}_val.txt'), 'w+') as f:
    #     for line in all_val:
    #         f.write(line + "\n")
    #
    # with open(os.path.join(save_dir, f'{name}_test.txt'), 'w+') as f:
    #     for line in all_test:
    #         f.write(line + "\n")


if __name__ == "__main__":
    # create_pose_dataset()
    # create_depth_dataset([46, 47, 48, 49, 50, 51, 52], name="depth_46to52")

    create_float_depth_dataset()
