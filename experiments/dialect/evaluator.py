import os
from pathlib import Path
from subprocess import TimeoutExpired

from typing import Dict
import pandas as pd
from pqdm.processes import pqdm
from sklearn import metrics
import pdb

from tqdm import tqdm


class Evaluator():

    def __init__(self,
                 data_dir: str,
                 sys_name: str,
                 experiment_dir: str,
                 dataset: str = "",
                 n_repetitions: int = 5,
                 n_workers = 64,
                 subset:int = None,
                 skip_processing = False,
                 augmented = False,
                 original = False,
                 ):

        self.n_repetitions = n_repetitions
        self.subset = subset
        self.data_dir = data_dir
        self.in_dir = data_dir + "/csv"
        self.dialect_file = data_dir + "/dialect_annotations.csv"
        self.skip_processing = skip_processing
        self.augmented = augmented
        self.original = original
        self.n_workers = n_workers
        if (augmented and original):
            self.sset = "complete"
        elif (not augmented) and original:
            self.sset = "original"
        elif (not original) and augmented:
            self.sset = "augmented"
        else:
            self.sset = "none"

        self.dataset = dataset
        self.experiment_dir = experiment_dir 
        self.sys_name = sys_name

        files = list(sorted(os.listdir(self.in_dir)))
        if self.augmented and self.original:
            pass
        elif self.augmented and not self.original:
            files = [f for f in files if "augmented" in f]
        elif self.original and not self.augmented:
            files = [f for f in files if "augmented" not in f]
        else:
            raise Exception("Invalid combination of augmented and original")
        assert len(files) > 0, "No files found"
        self.files = files[:self.subset]

    def process_file(self, in_filepath) -> Dict[str, str, str,float]:
        """

        :param in_filepath: a string containing the full path of the file to process
        :return a dictionary containing the keys "predicted_delimiter", "predicted_quotechar", "predicted_escapechar", "prediction_time"
        """
        raise NotImplementedError


    def print_results(self):
        results_file = self.experiment_dir + f"/{self.sys_name}/{self.sys_name}_{self.dataset}_{self.sset}_results_0.csv"
        results_df = pd.read_csv(results_file).fillna("").set_index("filename")
        annotations_df = pd.read_csv(self.dialect_file).fillna("").set_index("filename")
        metric_df = pd.merge(results_df, annotations_df, left_index=True, right_index=True)
        if len(results_df) != len(metric_df):
            pdb.set_trace()
            raise AssertionError("Error in merging")

        self.del_p, self.del_r, self.del_f1, self.del_s = metrics.precision_recall_fscore_support(metric_df["delimiter"], metric_df["predicted_delimiter"], average="micro", zero_division=0)
        self.quo_p, self.quo_r, self.quo_f1, self.quo_s = metrics.precision_recall_fscore_support(metric_df["quotechar"], metric_df["predicted_quotechar"], average="micro", zero_division=0)
        self.esc_p, self.esc_r, self.esc_f1, self.esc_s = metrics.precision_recall_fscore_support(metric_df["escapechar"], metric_df["predicted_escapechar"], average="micro", zero_division=0)


        metric_df["acc_dialect"] = (metric_df["delimiter"] == metric_df["predicted_delimiter"]) & \
                            (metric_df["quotechar"] == metric_df["predicted_quotechar"]) & \
                            (metric_df["escapechar"] == metric_df["predicted_escapechar"])

        self.accuracy =  metric_df["acc_dialect"].sum() / len(metric_df)

        self.avg_time = metric_df["prediction_time"].mean()
        self.avg_time_std = metric_df["prediction_time"].std()

        print("Results:")
        print("Delimiter F1", self.del_f1)
        print("Quotechar F1", self.quo_f1)
        print("Escapechar F1", self.esc_f1)
        print("Dialect Accuracy", self.accuracy)
        print("Average prediction time", self.avg_time, "+-", self.avg_time_std)

    def process_wrapper(self):
        args = [{"in_filepath": self.in_dir + "/" + f} for f in sorted(self.files)]
        if self.n_workers > 1:
            return pqdm(args, self.process_file, n_jobs=self.n_workers, argument_type='kwargs', desc="Processing files")
        else:
            return [self.process_file(**arg) for arg in tqdm(args, desc="Processing files")]

    def evaluate(self, *args, **kwargs):
        if self.skip_processing and os.path.exists(self.results_file):
            print("Skipping processing, results file already exists")
        else:
            for rep in [-1]+list(range(self.n_repetitions)):
                res = self.process_wrapper()
                if rep == -1:
                    continue

                if len([x for x in res if type(x) != dict]):
                    errors = [str(x) for x in res if type(x) != dict]
                    print("Error in processing file \n", "\n".join(set(errors)))

                results_df = pd.DataFrame(res).fillna("").set_index("filename")
                results_file = self.experiment_dir + f"/{self.sys_name}/{self.sys_name}_{self.dataset}_{self.sset}_results_{rep}.csv"
                Path(results_file).parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(results_file)