"""
Demo MCP Server for DeepEval Testing
Install: pip install fastmcp uvicorn
Run: python demo_mcp_server.py
"""

from fastmcp import FastMCP
import uvicorn

mcp = FastMCP("GitHub Demo Server")

@mcp.tool()
def search_repositories(query: str, language: str = None, limit: int = 5) -> str:
    """Search GitHub repositories.
    
    Args:
        query: Search query string
        language: Filter by programming language (optional)
        limit: Maximum number of results (default: 5)
    """
    # Mock data for demo
    repos = [
        {"name": "awesome-python", "stars": 15000, "language": "Python", "description": "A curated list of Python frameworks"},
        {"name": "react", "stars": 200000, "language": "JavaScript", "description": "A JavaScript library for building user interfaces"},
        {"name": "tensorflow", "stars": 180000, "language": "Python", "description": "An Open Source Machine Learning Framework"},
        {"name": "vue", "stars": 190000, "language": "JavaScript", "description": "Progressive JavaScript Framework"},
        {"name": "django", "stars": 70000, "language": "Python", "description": "The Web framework for perfectionists with deadlines"},
    ]
    
    # Filter by language if specified
    if language:
        repos = [r for r in repos if r["language"].lower() == language.lower()]
    
    # Filter by query
    filtered = [r for r in repos if query.lower() in r["name"].lower() or query.lower() in r["description"].lower()]
    
    # Limit results
    filtered = filtered[:limit]
    
    if not filtered:
        return f"No repositories found for query: {query}"
    
    result = f"Found {len(filtered)} repositories:\n"
    for repo in filtered:
        result += f"\n- {repo['name']} ({repo['language']})\n"
        result += f"  ⭐ {repo['stars']} stars\n"
        result += f"  {repo['description']}\n"
    
    return result

@mcp.tool()
def get_user_info(username: str) -> str:
    """Get GitHub user information.
    
    Args:
        username: GitHub username
    """
    # Mock data
    users = {
        "octocat": {
            "name": "The Octocat",
            "bio": "GitHub mascot",
            "repos": 8,
            "followers": 5000,
            "location": "San Francisco"
        },
        "torvalds": {
            "name": "Linus Torvalds",
            "bio": "Creator of Linux and Git",
            "repos": 5,
            "followers": 150000,
            "location": "Portland, OR"
        }
    }
    
    user = users.get(username.lower())
    if not user:
        return f"User '{username}' not found (using demo data)"
    
    return f"""User: {username}
Name: {user['name']}
Bio: {user['bio']}
Location: {user['location']}
Public Repos: {user['repos']}
Followers: {user['followers']}"""

@mcp.tool()
def get_trending_repos(language: str = "all", since: str = "daily") -> str:
    """Get trending repositories.
    
    Args:
        language: Programming language filter (default: all)
        since: Time range - daily, weekly, monthly (default: daily)
    """
    trending = {
        "Python": ["fastapi-ml", "ai-toolkit", "data-science-utils"],
        "JavaScript": ["react-hooks-pro", "next-gen-framework", "web3-tools"],
        "all": ["awesome-ai", "dev-resources", "open-source-guide"]
    }
    
    repos = trending.get(language, trending["all"])
    
    result = f"Trending {language} repositories ({since}):\n"
    for i, repo in enumerate(repos, 1):
        result += f"\n{i}. {repo}\n"
        result += f"   ⭐ {1000 * i} stars today\n"
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Demo MCP Server for DeepEval Testing")
    print("=" * 60)
    print("Server URL: http://localhost:8000/github/mcp")
    print("\nAvailable tools:")
    print("  - search_repositories: Search GitHub repos")
    print("  - get_user_info: Get user information")
    print("  - get_trending_repos: Get trending repositories")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    # FastMCP uses http_app() method to get the ASGI application
    # Configure the path to match the expected endpoint /github/mcp
    # This returns the Starlette app instance that can be used with uvicorn
    app = mcp.http_app(path="/github/mcp")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
