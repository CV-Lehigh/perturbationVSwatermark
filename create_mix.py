"""Mix watermarked images at different strengths into one folder."""
import argparse
import os
import random
import shutil


def split_into_three(lst):
    random.shuffle(lst)
    n = len(lst) // 3
    return lst[:n], lst[n:2 * n], lst[2 * n:]


def copy_mix(src_dirs, out_dir, prefixes, names):
    os.makedirs(out_dir, exist_ok=True)
    for names_part, src, prefix in zip(names, src_dirs, prefixes):
        for name in names_part:
            shutil.copy(os.path.join(src, name), os.path.join(out_dir, f"{prefix}_{name}"))


def copy_gen_mix(src_dirs, out_dir, prefixes, name_groups):
    os.makedirs(out_dir, exist_ok=True)
    for names_part, src, prefix in zip(name_groups, src_dirs, prefixes):
        for p in names_part:
            for img in os.listdir(src):
                if p in img:
                    shutil.copy(os.path.join(src, img), os.path.join(out_dir, f"{prefix}_{img}"))


def main():
    parser = argparse.ArgumentParser(description="Mix three watermark-strength folders into one.")
    parser.add_argument("--input_dirs", nargs=3, required=True,
                        help="Three input image folders (e.g. wm_0.1 wm_0.3 wm_0.5)")
    parser.add_argument("--input_out", required=True, help="Output mixed input folder")
    parser.add_argument("--gen_dirs", nargs=3, default=None,
                        help="Optional matching generation folders")
    parser.add_argument("--gen_out", default=None, help="Output mixed generation folder")
    parser.add_argument("--prefixes", nargs=3, default=["0.1", "0.3", "0.5"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    img_names = os.listdir(args.input_dirs[0])
    parts = split_into_three(img_names)

    copy_mix(args.input_dirs, args.input_out, args.prefixes, parts)
    if args.gen_dirs and args.gen_out:
        copy_gen_mix(args.gen_dirs, args.gen_out, args.prefixes, parts)

    n_in = len(os.listdir(args.input_out))
    n_gen = len(os.listdir(args.gen_out)) if args.gen_out else 0
    print(n_in, n_gen)


if __name__ == "__main__":
    main()
