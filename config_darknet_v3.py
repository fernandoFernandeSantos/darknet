#!/usr/bin/python
import os
import ConfigParser
import sys
import copy

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

DATASETS = [
    # normal
     {'txt': 'caltech.pedestrians.1K.txt', 'gold': 'gold.caltech.1K.csv', 'mode': 'full'},
    # {'txt': 'urban.street.1.1K.txt', 'gold': 'gold.urban.street.1.1K.csv', 'mode': 'full'},
    # {'txt': 'voc.2012.1K.txt', 'gold': 'gold.voc.2012.1K.csv', 'mode': 'full'},
]

BINARY_NAME = "darknet_v3"
# SAVE_LAYER = [0, ]
USE_TENSOR_CORES = [0, 1]
# 0 - "none",  1 - "gemm", 2 - "smart_pooling", 3 - "l1", 4 - "l2", 5 - "trained_weights"}
ABFT = [0]  # , 2]
WEIGHTS = "yolov3.weights"
CFG = "yolov3.cfg"


def config(board, debug, download_data):
    print "Generating darknet v3 for CUDA, board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = BINARY_NAME
    data_path = install_dir + "data/darknet"
    bin_path = install_dir + "bin"
    src_darknet = install_dir + "src/cuda/" + benchmark_bin

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # change it for darknetv2
    generate = ["mkdir -p " + bin_path, "mkdir -p /var/radiation-benchmarks/data", "cd " + src_darknet,
                "make clean GPU=1", "make -j4 GPU=1  SAFE_MALLOC=1",
                "mv ./" + benchmark_bin + "  " + bin_path + "/"]
    execute = []

    # 0 - "none",  1 - "gemm", 2 - "smart_pooling", 3 - "l1", 4 - "l2", 5 - "trained_weights"}

    for i in DATASETS:
        for tc in USE_TENSOR_CORES:
            if (save_layer == 1 and i['mode'] == 'full') or (save_layer == 0 and i['mode'] == 'small'):
                continue

            gold = data_path + '/' + BINARY_NAME + '_tensor_cores_mode_' + str(tc) + '_' + i['gold']
            txt_list = install_dir + 'data/networks_img_list/' + i['txt']
            gen = {
                'bin': [
                    "sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} " + bin_path,
                    "/" + benchmark_bin],
                # 'e': [' -e ', 'yolo'],  # execution_type =
                'aa': ['test_radiation', ''],  # execution_model =
                'c': [' -c ', data_path + '/' + CFG],  # config_file =
                'w': [' -w ', data_path + '/' + WEIGHTS],  # weights =
                'n': [' -n ', '1'],
                # iterations =  #it is not so much, since each dataset have at least 10k of images
                'g': [' -g ', gold],  # base_caltech_out = base_voc_out = src_darknet
                'l': [' -l ', txt_list],
                't': [' -t ', tc]
            }

            exe = copy.deepcopy(gen)
            exe['n'][1] = 10000
            exe['g'][0] = ' -d '

            exe_save = copy.deepcopy(exe)
            exe_save['s'][1] = save_layer

            if abft == 0:
                generate.append(" ".join([''.join(map(str, value)) for key, value in gen.iteritems()]))

            execute.append(" ".join([''.join(map(str, value)) for key, value in exe.iteritems()]))

    generate.append("make clean GPU=1 SAFE_MALLOC=1")
    generate.append("make -C ../../include/")
    generate.append("make -j 4 GPU=1 SAFE_MALLOC=1 LOGS=1")
    generate.append("mv ./" + benchmark_bin + " " + bin_path + "/")

    execute_and_write_json_to_file(execute=execute, generate=generate, install_dir=install_dir,
                                   benchmark_bin=benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    download_data = False
    try:
        parameter = str(sys.argv[1:][0]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
        if parameter == "DOWNLOAD_DATA":
            download_data = True
    except:
        debug_mode = False
        download_data = False
    board, _ = discover_board()
    config(board=board, debug=debug_mode, download_data=download_data)
