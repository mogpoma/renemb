import os
import time

import clevercsv

from evaluator import Evaluator

class CleverEvaluator(Evaluator):
    def __init__(self, limit_rows=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit_rows = limit_rows

    def process_file(self, in_filepath):
        try:
            with open(in_filepath, newline='', encoding="utf8") as in_csvfile:
                file = "\n".join(in_csvfile.read().splitlines()[:self.limit_rows])
        except Exception as e:
            with open(in_filepath, newline='', encoding="latin-1") as in_csvfile:
                file = "\n".join(in_csvfile.read().splitlines()[:self.limit_rows])

        start = time.process_time()
        try:
            dialect = clevercsv.Sniffer().sniff(file)
        except Exception as e:
            print(f'Error processing file {in_filepath}: {e}')
            dialect = None
        end = time.process_time()-start

        if dialect is not None:
            return {"filename": os.path.basename(in_filepath),
                    "predicted_delimiter": dialect.delimiter,
                    "predicted_quotechar": dialect.quotechar,
                    "predicted_escapechar": dialect.escapechar
                    , "prediction_time": end}
        else:
            return {"filename": os.path.basename(in_filepath),
                    "predicted_delimiter": "[UNK]",
                    "predicted_quotechar": "[UNK]",
                    "predicted_escapechar": "[UNK]",
                    "prediction_time": end}


if __name__ == "__main__":
    sys = "clevercs_sample"
    limit = 128

    print("Test set:")
    test_evaluator = CleverEvaluator(data_dir = "data/dialect_detection/test/",
                                    sys_name=sys,
                                    experiment_dir = "results/dialect/",
                                    dataset = "test",
                                    subset=None,
                                    n_workers=100,
                                    augmented=False,
                                    original=True,
                                    limit_rows = limit,
                                    )
    test_evaluator.evaluate()
    test_evaluator.print_results()

    print("Difficult set:")
    test_evaluator = CleverEvaluator(data_dir = "data/dialect_detection/difficult/",
                                    sys_name=sys,
                                    experiment_dir = "results/dialect/",
                                    dataset = "difficult",
                                    subset=None,
                                    n_workers=100,
                                    augmented=False,
                                    original=True,
                                    limit_rows = limit,)
    test_evaluator.evaluate()
    test_evaluator.print_results()

    print("Overall set:")
    test_evaluator = CleverEvaluator(data_dir = "data/dialect_detection/overall/",
                                    sys_name="clevercs",
                                    experiment_dir = "experiments/finetune_dialect/",
                                    dataset = "overall",
                                    subset=None,
                                    n_workers=100,
                                    augmented=False,
                                    original=True)
    test_evaluator.evaluate()
    test_evaluator.print_results()