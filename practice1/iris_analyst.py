import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# 함수, 클래스, 전역변수








# 실행 코드 짜는 곳
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Cov = X.T @ X
    print(Cov)


