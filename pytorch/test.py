import subprocess
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', help='mode')
args = parser.parse_args()
if args.mode == "in":
    cmd = "./pal_loader ./python3.manifest ./label_att/src/tee/mnist_example_intee.py --batch 40 --mode train --epoch 10 --host 127.0.0.1 --port1 12344 --port2 12346 --pp 4"
    # cmd = "scp -P 2215 john@127.0.0.1:/home/john/label_att/fig/scal_accu_1.pdf fig/"
    process = subprocess.Popen(cmd.split())
    output, error = process.communicate()
elif args.mode == "out":
    cmd = "python3 label_att/src/tee/mnist_example_outtee.py --batch 40 --mode train --epoch 10 --host 127.0.0.1 --port1 12345 --port2 12346 --pp 4"
    # cmd = "scp -P 2215 john@127.0.0.1:/home/john/label_att/fig/scal_accu_1.pdf fig/"
    process = subprocess.Popen(cmd.split())
    output, error = process.communicate()
