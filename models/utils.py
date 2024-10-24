from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

def gdrive_service():
  """Google Drive authentication and service setup"""
  SCOPES = ["https://www.googleapis.com/auth/drive"]
  SERVICE_ACCOUNT_FILE = "./models/gdrive-service.json"

  credentials = Credentials.from_service_account_file(
      SERVICE_ACCOUNT_FILE, scopes=SCOPES)
  
  service = build("drive", "v3", credentials=credentials)
  return service

def create_and_share_folder(service, folder_name, user_emails, parent_id=None):
  """Create new folder in Google Drive (Cloud Service) and share it with specified users"""
  folder_metadata = {
      'name': folder_name,
      'mimeType': 'application/vnd.google-apps.folder',
      'parents': [parent_id] if parent_id else []
  }
  folder = service.files().create(body=folder_metadata, fields='id').execute()

  if user_emails:
      for user_email in user_emails:
          user_permission = {
              'type': 'user',
              'role': 'writer',
              'emailAddress': user_email
          }
          service.permissions().create(
              fileId=folder.get('id'),
              body=user_permission,
              fields='id',
          ).execute()

  return folder.get('id')

def create_subfolder(service, parent_folder_id, subfolder_name):
  """Create a subfolder within a given parent folder."""
  subfolder_metadata = {
      'name': subfolder_name,
      'mimeType': 'application/vnd.google-apps.folder',
      'parents': [parent_folder_id]
  }
  subfolder = service.files().create(body=subfolder_metadata, fields='id').execute()
  return subfolder.get('id')
  