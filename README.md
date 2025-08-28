Job Poster
===========

This script polls a WordPress REST API for new posts and posts them to a Telegram channel after summarizing with the Gemini API.

Usage
-----

Set environment variables:

- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHANNEL_ID
- GEMINI_API_KEY
  
Optional (to use MongoDB for tracking posted jobs instead of local file):

- MONGODB_URI (e.g., `mongodb+srv://user:pass@cluster.example.com/?retryWrites=true&w=majority`)
- MONGODB_DB (default: `jobposter`)
- MONGODB_COLLECTION (default: `posted_jobs`)

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run with live API:

```powershell
python job_poster.py
```

Run with the provided sample `res.json` for testing:

```powershell
python job_poster.py --use-sample
```

Notes
-----
- The script stores posted post IDs in `posted.json` next to the script.
- If `MONGODB_URI` is provided (and `pymongo` is installed), posted IDs will be stored in MongoDB collection instead. Each posted job is stored as a document with `_id` equal to the WordPress post id.
- If Gemini client is not installed or the API call fails, posts are skipped and errors are logged to console.
