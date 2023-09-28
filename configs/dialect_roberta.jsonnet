local max_rows = 128;
local max_len = 128;
local seed = 42;
local n_files = 20000;
local num_workers=1;
local train_dataset = "train_augmented";
local dev_dataset = "dev_augmented";
local test_dataset = "test_augmented";
local max_len = 510;
local experiment = "roberta_dialect";

{
  data_module: {
  train_data_path: "data/dialect_detection/"+train_dataset,
  val_data_path : "data/dialect_detection/"+dev_dataset,
  test_data_path : "data/dialect_detection/"+test_dataset,
  save_path : "results/dialect/roberta/",
  "batch_size" : 16,
  "num_workers": num_workers,
  "max_rows": max_rows,
  "n_files":n_files,
  "max_len":max_len,
  },
 vocabulary: {
      directory:"vocabulary/dialect_detection",
  },
   model: {
    optimizer_lr:1e-4,
    save_path : "weights/roberta_dialect.pth",
  },
    trainer:{
   default_root_dir: "results/dialect/",
    accelerator: "gpu",
    devices:[0],
    min_epochs:1,
    max_epochs:10,
    precision:"16-mixed",
    log_every_n_steps:10,
    },
    logger:{
     save_dir:"results/dialect/tensorboard/",
     version: "roberta",
    },
    callbacks: {
    save_magritte_model:{},
    early_stopping:{"monitor":"val_loss", "patience":5},
    },
}
