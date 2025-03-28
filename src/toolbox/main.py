"""Entry point for the toolbox package."""

from pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline(
        "/home/adamwa/Ocean Informatics/Projects/toolbox/src/toolbox/config.yaml"
    )
    pipeline.run()
    pipeline.visualise_pipeline()
