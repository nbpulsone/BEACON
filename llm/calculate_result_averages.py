import numpy
import os
import sys
import os


def calculate_weights(input_dir):
  # calculate result weights
  weights = {}
  sum_lens = 0
  for filename in sorted(os.listdir(input_dir)):
      if filename.endswith("_test.txt"):
          with open(os.path.join(input_dir, filename), "r") as f:
              lines = f.readlines()
              weights[filename[:-9]] = len(lines)
              sum_lens += len(lines)
  for w in weights:
      weights[w] /= sum_lens
  return weights


if __name__ == "__main__":
    # parse input
    if len(sys.argv) < 3:
        print(f"Usage: python calculate_result_averages.py <result_file> <input_dir_with_test_files>")
    RESULT_FILE = sys.argv[1]
    INPUT_DIR = sys.argv[2]

    # calcualte result weights
    category_weights = calculate_weights(INPUT_DIR)

    # parse result file
    f1_scores = {}
    precisions = {}
    recalls = {}
    with open(RESULT_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 2 and not line.startswith("~~~"):
                t = line.split(' ')
                if len(t) >= 3:
                    domain = t[0][:-9]
                    metric = t[1]
                    score = float(t[2])
                    if metric.startswith('F1'):
                        f1_scores[domain] = score
                    elif metric.startswith('Prec'):
                        precisions[domain] = score
                    else:
                        recalls[domain] = score

    # calculate averages
    f1_tot = 0
    f1_tot_w = 0
    sum_w = 0
    for dom in f1_scores:
        f1_tot += f1_scores[dom]
        f1_tot_w += f1_scores[dom] * category_weights[dom]
        sum_w += category_weights[dom]

    # report averages
    print(f"{INPUT_DIR}: ")
    print(f"Average F1 Score: {f1_tot/len(f1_scores)}")
    print(f"Weighted F1 Score: {f1_tot_w/sum_w}")
