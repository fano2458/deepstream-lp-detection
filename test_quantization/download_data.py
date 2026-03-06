from roboflow import Roboflow


rf = Roboflow(api_key="...")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(12)
dataset = version.download("yolo26")
                