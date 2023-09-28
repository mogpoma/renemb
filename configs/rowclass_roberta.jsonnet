local n_files = 50000;
local max_rows = 128;
local dataset = 'gittables';
local d_model = 768;
local max_len = 128;
local num_workers = 100;
local validation_dataset = std.extVar('validation_dataset');

{
  data_module: {
   shuffle:true,
   batch_size : 16,
   num_workers: num_workers,
   max_rows: max_rows,
   max_len: max_len,
   n_files: n_files,
  save_path : "results/rowclass/roberta/",
  data_path: "data/row_classification/strudel_annotations_cv.jsonl",
  train_datasets: ["cv_0", "cv_1", "cv_2", "cv_3", "cv_4", "cv_5", "cv_6", "cv_7", "cv_8", "cv_9"],
  val_dataset_name: validation_dataset,
  },
  vocabulary: {
      type:"from_files",
      directory:"vocabulary/"+dataset,
  },
  model: {
    n_classes:6,
    ignore_class:"empty",
    classes_weights: [0.01,1, 1, 1, 1, 1],
    optimizer_lr:1e-5,
    save_path:"weights/rowclass/roberta_rowclass_"+validation_dataset+".pth"
  },
    trainer:{
    accelerator: "gpu",
    devices:[0],
    precision:"16-mixed",
    min_epochs: 1,
    max_epochs: 20,
    log_every_n_steps:10,
    },
    logger:{
     save_dir:"results/rowclass/tensorboard/",
     version: "roberta_val_"+validation_dataset,
    },
    callbacks: {
    early_stopping:{"monitor":"val_loss", "patience":5},
    },
}