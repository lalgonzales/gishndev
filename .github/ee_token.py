import os
import json

ee_token = os.environ["EARTHENGINE_TOKEN"]
credential = {"refresh_token": ee_token}
credential_file_path = os.path.expanduser("~/.config/earthengine/")
os.makedirs(credential_file_path, exist_ok=True)
with open(os.path.join(credential_file_path, "credentials"), "w", encoding="utf-8") as file:
    json.dump(credential, file)
