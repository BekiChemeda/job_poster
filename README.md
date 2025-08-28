Job Poster
===========

This script polls a WordPress REST API for new posts and posts them to a Telegram channel after summarizing with the Gemini API.

Usage
-----

Set environment variables:

- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHANNEL_ID
- GEMINI_API_KEY

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
- If Gemini client is not installed or the API call fails, posts are skipped and errors are logged to console.
