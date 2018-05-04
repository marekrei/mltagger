import sys


from model import MLTModel
from evaluator import MLTEvaluator
from experiment import read_input_files


if __name__ == "__main__":
    model = MLTModel.load(sys.argv[1])
    data = read_input_files(sys.argv[2], -1)
    batch_size = 32
    evaluator = MLTEvaluator(model.config)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        cost, sentence_scores, token_scores_list = model.process_batch(batch, False, 0.0)

        for j in range(len(batch)):
            for k in range(len(batch[j])):
                print(" ".join([str(x) for x in batch[j][k]]) + "\t" + str(token_scores_list[0][j][k]) + "\t" + str(sentence_scores[j]))
            print("")

        evaluator.append_data(cost, batch, sentence_scores, token_scores_list)

    results = evaluator.get_results("test")
    for key in results:
        sys.stderr.write(key + ": " + str(results[key]) + "\n")
