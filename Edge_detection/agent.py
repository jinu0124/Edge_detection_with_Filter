import os
from edge_detection import edge

# 2015253039 권진우 영상처리 HW3
if __name__ == '__main__':
    ROOT_DIR = os.path.abspath("")
    MODEL_PATH = os.path.join(ROOT_DIR)
    edge(MODEL_PATH)