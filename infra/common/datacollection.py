import os
import json
import base64
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from domain import utils
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import josephlogging.log as log

logger = log.getLogger(__name__)

# Scope for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']
root_id = "1HHmaIGH7k90x1c58GekiRdyopoh02xI5"

client = None

# Initialize a ThreadPoolExecutor with a fixed number of worker threads
executor = ThreadPoolExecutor(max_workers=5)  # Adjust max_workers as needed

def init():
    global client
    # Load credentials from environment variable
    GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')

    if not GOOGLE_CREDENTIALS_JSON:
        raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set")

    # Parse the JSON credentials
    try:
        credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)
        logger.info(f"Using service account: {credentials_info.get('client_email')}")
        credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON in GOOGLE_CREDENTIALS_JSON") from e
    
    return build('drive', 'v3', credentials=credentials)

def create_folder(service, folder_name, parent_folder_id=None):
    lst = list_folder(service, parent_folder_id)
    for k, v in lst.items():
        if v == folder_name:
            # logger.info(f'Folder "{folder_name}" already exists with ID: {k}')
            return k

    """Create a folder in Google Drive and return its ID."""
    folder_metadata = {
        'name': folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        'parents': [parent_folder_id] if parent_folder_id else []
    }
    created_folder = service.files().create(
        body=folder_metadata,
        fields='id'
    ).execute()

    # logger.info(f'Created Folder "{folder_name}" with ID: {created_folder["id"]}')
    return created_folder["id"]

def list_folder(service, parent_folder_id=None):
    """List folders and files in Google Drive."""
    query = f"'{parent_folder_id}' in parents and trashed=false" if parent_folder_id else "trashed=false"
    results = service.files().list(
        q=query,
        pageSize=1000,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    items = results.get('files', [])

    if not items:
        return {}
    else:
        itm = {}
        for item in items:
            itm[item['id']] = item['name']
        return itm

def get_data_name(timestamp:int, task:str, model:str, score:float, extra:str='0'):
    str_score = '{:.3f}'.format(score).replace('.','d')
    date = str(datetime.date.today())
    name = '_'.join([str(timestamp), task, model, str_score])
    filename = '_'.join([str(timestamp), task, model, str_score, extra])
    return filename, os.path.join(task, date, name, filename)

def upload(localpath: str, filename: str, metadata: str):
    """Upload a file to Google Drive using service account.
    
    Args:
        localpath: Path to the local file
        filename: Name to be used in Google Drive
        metadata: Additional metadata to be stored
    
    Returns:
        dict: Response from Google Drive API
    """
    try:
        # Get service using service account credentials
        service = init()
        
        # Create folder using date
        date = str(datetime.date.today())
        folder_id = create_folder(service, date, root_id)
    
        if not folder_id:
            raise Exception("Invalid folder ID")

        # Prepare the file metadata
        file_metadata = {
            'name': filename,
            'parents': [folder_id],
            'description': metadata
        }
        
        # Create media
        media = MediaFileUpload(localpath, resumable=True)
    
        # Execute the upload
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
    
        logger.info(f"File uploaded successfully. File ID: {file.get('id')}")
        
    except Exception as e:
        logger.error(f"Error uploading file '{filename}': {e}")
        raise
    finally:
        # Clean up the local file if it exists
        if os.path.exists(localpath):
            try:
                os.remove(localpath)
                # logger.info(f"Removed local file: {localpath}")
            except Exception as e:
                logger.warning(f"Could not remove local file '{localpath}': {e}")

def upload_async(localpath: str, filename: str, metadata: str):
    """Asynchronously upload a file to Google Drive."""
    future = executor.submit(upload, localpath, filename, metadata)
    return future

def upload_frame(timestamp:int, img:str, task:str, model:str, metadata:str, score:float, extra:str='0', ext='jpg', b64=True):
    name, fullname = get_data_name(timestamp, task, model, score, extra) 
    if b64:
        img2, ext = utils.trim_base64_header(logger, img)
    else:
        img2 = img
        ext = 'jpg'
    name = name + f'.{ext}' 
    fullname = fullname + f'.{ext}'
    path = os.path.join(os.getcwd(), 'infra/common/datalogs', name)
    try:
        with open(path, 'wb+') as fh:
            fh.write(base64.decodebytes(bytes(img2, 'utf-8')))
        #logger.info(f"Image saved locally at: {path}")
        # Initiate asynchronous upload
        future = upload_async(localpath=path, filename=fullname, metadata=metadata)
        #logger.info(f"Started asynchronous upload for file: {fullname}")
        return future
    except Exception as e:
        logger.error(f"Error in upload_frame: {e}")
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.warning(f"Removed local file due to error: {path}")
            except Exception as ex:
                logger.warning(f"Could not remove local file '{path}': {ex}")
        raise

if __name__ == '__main__':
    path = 'test_img/01.jpg'
    with open(path, 'rb') as fh:
        img = base64.b64encode(fh.read()).decode('utf-8')
    future = upload_frame(0, img, 'task', 'model', 'metadata', 0.0, 'extra', 'jpg', b64=True)
    # Optionally, wait for the upload to complete (not recommended if you want it to be non-blocking)
    # result = future.result()
    print("Upload initiated asynchronously.")

