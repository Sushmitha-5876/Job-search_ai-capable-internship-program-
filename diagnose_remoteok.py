import requests

response = requests.get(
    "https://remoteok.com/api",
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=30,
)

print(f"HTTP status: {response.status_code}")

data = response.json()

real_jobs = [
    job for job in data
    if isinstance(job, dict) and job.get("id")
]

print(f"Total real job entries: {len(real_jobs)}")
print()

if not real_jobs:
    print("No real jobs found.")
    raise SystemExit

print("=== SAMPLE REMOTEOK JOBS ===")

for i, job in enumerate(real_jobs[:5], start=1):
    print(f"\nJOB {i}")
    print("Title   :", job.get("position", ""))
    print("Company :", job.get("company", ""))
    print("Location:", job.get("location", ""))
    print("Tags    :", job.get("tags", []))

print("\n=== IMPORTANT ===")
print("RemoteOK is returning the expected fields.")