from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
c = MlflowClient()
RUN = "b1b2935fc163465eaf8ded21d86b6655"      # effnet_b0


def walk(path=None, depth=0):
    for f in c.list_artifacts(RUN, path):
        print("  " * depth + f.path + ("/" if f.is_dir else f"  ({f.file_size} bytes)"))
        if f.is_dir:
            walk(f.path, depth + 1)


print("--- artifacts inside the run ---")
walk()

run = c.get_run(RUN)
print("\nartifact_uri:", run.info.artifact_uri)
try:
    print("run outputs:", run.outputs)
except Exception as e:
    print("run outputs unavailable:", repr(e)[:150])

print("\n--- logged models in this experiment ---")
try:
    for m in c.search_logged_models(experiment_ids=[run.info.experiment_id]):
        print(m.model_id, m.name, m.artifact_location)
except Exception as e:
    print("search_logged_models failed:", repr(e)[:200])