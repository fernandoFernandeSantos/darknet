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

BINARY_NAME = "darknet_v3_"
# SAVE_LAYER = [0, ]
USE_TENSOR_CORES = [0, 1]
# 0 - "none",  1 - "gemm", 2 - "smart_pooling", 3 - "l1", 4 - "l2", 5 - "trained_weights"}
ABFT = [0]  # , 2]
REAL_TYPES = ["double", "single", "half"]
WEIGHTS = "yolov3.weights"
CFG = "yolov3.cfg"


def config(board, debug):
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
                "make clean GPU=1", "make -C ../../include/"]
    execute = []

    # 0 - "none",  1 - "gemm", 2 - "smart_pooling", 3 - "l1", 4 - "l2", 5 - "trained_weights"}
    for fp_precision in REAL_TYPES:
        for i in DATASETS:
            for tc in USE_TENSOR_CORES:
                generate.append("make clean GPU=1 LOGS=1")
                bin_final_name = benchmark_bin + fp_precision
                generate.append("make -j4 GPU=1 REAL_TYPE=" + fp_precision)
                generate.append("mv ./" + bin_final_name + "  " + bin_path + "/")

                gold = data_path + '/' + BINARY_NAME + 'tensor_cores_mode_' + str(tc) + '_fp_precision_' + str(
                    fp_precision) + '_' + i['gold']
                txt_list = install_dir + 'data/networks_img_list/' + i['txt']
                gen = [] * 8
                gen[0] = [
                    "sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} " + bin_path,
                    "/" + bin_final_name]
                gen[1] = [" detector ", " test_radiation "]
                gen[2] = [data_path + '/' + CFG]
                gen[3] = [data_path + '/' + WEIGHTS]
                gen[4] = [txt_list]
                gen[5] = [' -generate ', '1']
                gen[6] = [' -iterations ', '1']
                gen[7] = [' -tensor_cores ', tc]
                gen[8] = [' -gold ', gold]

                generate.append("make -j 4 GPU=1 LOGS=1 REAL_TYPE=" + fp_precision)
                generate.append("mv ./" + bin_final_name + "  " + bin_path + "/")

                exe = copy.deepcopy(gen)
                exe[6][1] = '1000000'
                exe[5][1] = '0'

                generate.append(" ".join([''.join(value) for value in gen]))

                execute.append(" ".join([''.join(value) for value in exe]))

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
    config(board=board, debug=debug_mode)
