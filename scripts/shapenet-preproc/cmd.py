#!/usr/bin/env python

# modified from https://gist.github.com/awesomebytes/a3bc8729d0c1d0a9499172b9a77d2622
import argparse
import os
import subprocess
from time import strftime
from zipfile import ZipFile
from subprocess import STDOUT

from tqdm import tqdm

try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    import os

    DEVNULL = open(os.devnull, 'wb')

from scripts.filters import filter_script_mlx

cwd = os.getcwd()

command = "{binary} -i {in_file} " \
          "-o {out_file} -m {thing_to_safe} " \
          "-s {filter_file}"


def create_filter_file(filename='filter'):
    # filen_ = os.path.join(tempfile.gettempdir(),"{}-{}.mlx".format(
    #         filename, strftime("%y%m%d-%H%M%S")))
    filen_ = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tmp",
        "{}-{}.mlx".format(filename, strftime("%y%m%d-%H%M%S")))
    with open(filen_, 'w') as f:
        f.write(filter_script_mlx[1:])
    return filen_


def reduce_faces(in_file, out_file, filter_file):
    command_inst = command.format(
        binary="meshlab.meshlabserver",
        in_file=in_file,
        out_file=out_file,
        thing_to_safe=" ".join([
            "fn"  # face normals
        ]),
        filter_file=filter_file
    )

    # print("Going to execute: " + command_inst)

    try:
        subprocess.check_output(command_inst, shell=True, stderr=STDOUT)
    except subprocess.CalledProcessError:
        print ("Error processing file. "
               "Remove the `stderr=STDOUT` "
               "in the line above this "
               "exception and try again")


def read_args():
    parser = argparse.ArgumentParser(description='IG-Generator')
    parser.add_argument('--path', required=True,
                        help='path to the folder with all the model zip files')

    args = parser.parse_args()
    # print(args)
    return args


def is_zip_file(f):
    return os.path.isfile(f) and f[-4:] == ".zip"


def is_model_file(f):
    return os.path.isdir(f) and os.path.isfile(os.path.join(f, "model.dae"))


if __name__ == '__main__':
    args = read_args()

    zip_files = [f for f in os.listdir(args.path)
                 if is_zip_file(os.path.join(args.path, f))]

    for z in tqdm(zip_files):
        if not os.path.isfile(os.path.join(args.path, z[:-4], "model.dae")):
            with ZipFile(os.path.join(args.path, z), "r") as zip_ref:
                zip_ref.extractall(os.path.join(args.path, z[:-4]))

    model_files = [f for f in os.listdir(args.path)
                   if is_model_file(os.path.join(args.path, f))]

    filter_file = create_filter_file()

    output_path = os.path.join(args.path, "_out")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for m in tqdm(model_files):
        # print(m)

        reduce_faces(
            in_file=os.path.join(args.path, m, "model.dae"),
            out_file=os.path.join(output_path, m.lower() + ".dae"),
            filter_file=filter_file
        )
