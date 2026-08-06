"""
Fetches real issues and pull requests from a real GitHub repository as
RAW MATERIAL for building a manually curated real-repo benchmark
(experiment E6) — this does NOT auto-generate tasks.
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.error

DECISION_KEYWORDS = [
    "instead of", "decided", "decision", "rejected", "switching from",
    "switched from", "reverted", "revert", "deprecat", "migrat",
    "instead", "rather than", "no longer", "moved away from",
]


def _github_request(url: str, token: str = None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"GitHub API error {e.code}: {body}") from e


def fetch_issues(repo: str, token: str, limit: int, min_comments: int) -> list:
    collected = []
    page = 1
    per_page = 100
    while len(collected) < limit:
        url = (
            f"https://api.github.com/repos/{repo}/issues"
            f"?state=closed&sort=comments&direction=desc&per_page={per_page}&page={page}"
        )
        batch = _github_request(url, token)
        if not batch:
            break
        for issue in batch:
            if "pull_request" in issue:
                continue
            if issue.get("comments", 0) < min_comments:
                continue
            collected.append(issue)
        page += 1
        time.sleep(0.3)
        if len(batch) < per_page:
            break
    return collected[:limit]


def flag_decision_candidates(issues: list) -> list:
    flagged = []
    for issue in issues:
        text = f"{issue.get('title', '')} {issue.get('body') or ''}".lower()
        matched_keywords = [kw for kw in DECISION_KEYWORDS if kw in text]
        if matched_keywords:
            flagged.append({
                "number": issue["number"],
                "title": issue["title"],
                "url": issue["html_url"],
                "comments": issue["comments"],
                "matched_keywords": matched_keywords,
                "body_excerpt": (issue.get("body") or "")[:500],
            })
    return flagged


def main():
    parser = argparse.ArgumentParser(description="Fetch real GitHub issues as candidate material for manual E6 task curation")
    parser.add_argument("--repo", required=True, help="owner/name, e.g. expressjs/express")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-comments", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Warning: no GITHUB_TOKEN set — limited to 60 requests/hour, may fail on a busy IP.")
        print("Get a free token at https://github.com/settings/tokens (no special scopes needed for public repos).")

    print(f"Fetching issues from {args.repo}...")
    issues = fetch_issues(args.repo, token, args.limit, args.min_comments)
    print(f"Fetched {len(issues)} issues with >= {args.min_comments} comments.")

    candidates = flag_decision_candidates(issues)
    print(f"{len(candidates)} flagged as containing decision-indicating language.")

    output_path = args.output or f"candidates_{args.repo.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"\nCandidates written to {output_path}")
    print("Next step: manually review this file and curate real task pairs.")


if __name__ == "__main__":
    main()