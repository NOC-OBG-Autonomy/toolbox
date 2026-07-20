from pelagos_py.pipeline import Pipeline

# Point the pipeline at a config file and run it.
p = Pipeline(config_path="examples/configs/complete_config.yaml")
p.run()
