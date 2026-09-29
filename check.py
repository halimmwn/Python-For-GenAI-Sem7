import httpx
import os
from dotenv import load_dotenv

load_dotenv()
res = httpx.get('https://api.xkiro.com/v1/models', headers={'Authorization': 'Bearer ' + os.environ['XKIRO_API_KEY']})
print([m['id'] for m in res.json().get('data', []) if 'claude' in m['id'].lower()])
