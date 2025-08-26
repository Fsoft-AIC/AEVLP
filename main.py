import os
import traceback
from config import get_configs
from train import run_train
import json

def main():
    P = get_configs()
    # save config
    with open(f"{P['save_path']}/config.json", 'w') as f:
        json.dump(P, f, indent=4)
    print(P, '\n')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    print('###### Train start ######')
    run_train(P)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
