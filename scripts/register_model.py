from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()                      # DagsHub address + token from .env
NAME = "VisionGuard"
RUNS = {                           # order matters: baseline first, champion second
    "manual_baseline": "bca0edeb15d74ee49431eb130ef73fe6",
    "effnet_b0": "b1b2935fc163465eaf8ded21d86b6655",
}

c = MlflowClient()

# remove the earlier registration, which pointed at the wrong folder
try:
    c.delete_registered_model(NAME)
    print("deleted the old registration")
except Exception as e:
    print("could not delete old registration (continuing):", repr(e)[:120])
try:
    c.create_registered_model(NAME)
except Exception:
    pass                           # still exists, new versions will just get higher numbers

exp_id = c.get_run(RUNS["effnet_b0"]).info.experiment_id
location = {m.model_id: m.artifact_location
            for m in c.search_logged_models(experiment_ids=[exp_id])}

versions = {}
for label, run_id in RUNS.items():
    model_id = c.get_run(run_id).outputs.model_outputs[0].model_id
    source = location[model_id]
    mv = c.create_model_version(NAME, source=source, run_id=run_id)
    versions[label] = mv.version
    print(f"{label}: registered as version {mv.version}  (model {model_id})")
    try:
        c.set_model_version_tag(NAME, mv.version, "source_run", label)
    except Exception as e:
        print("  (tag not set:", repr(e)[:120], ")")

c.set_registered_model_alias(NAME, "champion", versions["effnet_b0"])
mv = c.get_model_version_by_alias(NAME, "champion")
print(f"alias 'champion' -> version {mv.version} (run {mv.run_id})")
print("source:", mv.source)