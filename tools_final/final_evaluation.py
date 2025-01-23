import pandas as pd
import numpy as np

def compute_area(data, thresholds):
    size = data.shape[0]
    results=[]

    for threshold in thresholds:
        results.append((data < threshold).astype(int).sum() / size)

    return np.array(results)

def compute_auc(data, thresholds):
    size = data.shape[0]
    results = []

    for threshold in thresholds:
        temp = ((data[:, 0] < threshold) & (data[:, 1] < threshold)).astype(int).sum() / size
        results.append(temp)

    return np.array(results)

def final_evaluation(path="/home/washindeiru/studia/7_semestr/vo/comparison_final", name="superglue.txt"):
    df = pd.read_csv(path + "/" + name, sep=" ", header=None, comment="#")
    matrix = df.to_numpy()

    thresholds = [5, 10, 20, 30]

    #auc
    auc_result = compute_auc(matrix, thresholds)
    return auc_result.reshape(-1, 1)

def make_evaluation(path):
    filenames = ['superglue.txt', 'lightglue.txt', 'gluestick.txt', 'tartanvo.txt', 'devo.txt']

    result = None

    for name in filenames:
        temp = final_evaluation(path=path, name=name)
        if result is None:
            result = temp
        else:
            result = np.concatenate((result, temp), axis=1)

    df = pd.DataFrame(result, columns=filenames)

    path=path + "/final_evaluation.txt"
    comment = "# 5 10 20 30"
    with open(path, 'w') as f:
        f.write(comment + "\n")
        df.to_csv(f, index=False)


if __name__ == "__main__":
    path = "/home/washindeiru/studia/7_semestr/vo/comparison_final_9"
    make_evaluation(path)