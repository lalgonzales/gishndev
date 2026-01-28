import ee
import json
import os
import google.oauth2.credentials

stored = json.loads(os.getenv("EARTHENGINE_TOKEN"))
credentials = google.oauth2.credentials.Credentials(
    None,
    token_uri="https://oauth2.googleapis.com/token",
    client_id=stored["client_id"],
    client_secret=stored["client_secret"],
    refresh_token=stored["refresh_token"],
    quota_project_id=stored["project"],
)
credential_file_path = os.path.expanduser("~/.config/earthengine/")
os.makedirs(credential_file_path, exist_ok=True)
with open(credential_file_path + "credentials", "w") as file:
    file.write(credentials.to_json())

ee.Initialize(credentials=credentials)

print(ee.String("Greetings from the Earth Engine servers!").getInfo())
