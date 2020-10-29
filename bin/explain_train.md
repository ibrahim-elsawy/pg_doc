#Explaintion of train.py
*  The block below used to parse the cmd options of parameters, it uses flags module in tensorflow similar to pyparse library
```python
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("master", "",
                    "Master tensorflow server (empty string for local).")
flags.DEFINE_string("model_dir", None,
                    "Output directory for model checkpoints.")
flags.DEFINE_string("params", None, "Name of params to use.")
flags.DEFINE_string("param_overrides", None,
                    "Param overrides as key=value pairs")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "Number of iterations to perform per TPU loop.")
flags.DEFINE_integer("num_shards", 1, "Number of TPU shards available.")
flags.DEFINE_boolean("use_tpu", False, "Whether to run on TPU accelerators.")
flags.DEFINE_integer("train_infeed_parallelism", 32,
                     "Number of infeed threads for training.")
flags.DEFINE_string("train_init_checkpoint", None,
                    "Initialize model or partial model from this checkpoint.")
flags.DEFINE_integer("train_warmup_steps", 10000, "Number of steps to warmup.")
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "Save checkpoints every this many steps.")
flags.DEFINE_integer(
    "keep_checkpoint_max", 5,
    "The maximum number of recent checkpoint files to keep. "
    "As new files are created, older files are deleted.")
flags.DEFINE_string("train_steps_overrides", "",
                    ("List of integers. Override train steps from params."
                     "Ensure that model is saved at specified train steps."))
flags.DEFINE_integer("tfds_train_examples", -1,
                     "Set number of examples for tfds type data source")
``` 
* registery.get_params


* estimator_utils.create_estimator

```python
  estimator = estimator_utils.create_estimator(
      FLAGS.master,
      FLAGS.model_dir,
      FLAGS.use_tpu,
      FLAGS.iterations_per_loop,
      FLAGS.num_shards,
      params,
      train_init_checkpoint=FLAGS.train_init_checkpoint,
      train_warmup_steps=FLAGS.train_warmup_steps,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)
```

* estimator_utils.train
```python
    estimator.train(
        input_fn=infeed.get_input_fn(
            params.parser,
            params.train_pattern,
            tf.estimator.ModeKeys.TRAIN,
            parallelism=FLAGS.train_infeed_parallelism),
        max_steps=train_steps)
```

