import streamlit as st
import openai
import os
import requests
import re
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import Counter
import math
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Configure OpenAI API
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY! Please set it in your environment variables.")
    st.stop()

openai.api_key = OPENAI_API_KEY

# GitHub API Headers
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Extract code blocks from AI response
def extract_code(text):
    """Extracts code block from markdown text."""
    match = re.search(r"```(?:\w+)?\n([\s\S]*?)\n```", text)
    return match.group(1).strip() if match else None

# Extract suggested filename
def extract_filename(text):
    """Extracts filename from AI response."""
    match = re.search(r"Suggested Filename:\s*`?([\w.-]+)`?", text, re.IGNORECASE)
    return match.group(1).strip() if match else "unknown_file.txt"

# Fetch repository details from GitHub
def fetch_github_repo(repo_url):
    """Fetches repository details including README and file list."""
    if not GITHUB_TOKEN:
        return None, None, "❌ Missing GitHub API Token!"

    # Extract owner and repo name from URL
    parts = repo_url.strip("/").split("/")
    if len(parts) < 2:
        return None, None, "❌ Invalid GitHub repository URL!"

    owner, repo = parts[-2], parts[-1]
    repo_api_url = f"https://api.github.com/repos/{owner}/{repo}"
    
    # Fetch repo details
    repo_response = requests.get(repo_api_url, headers=HEADERS)
    if repo_response.status_code != 200:
        return None, None, "❌ Repository not found or access denied!"

    repo_data = repo_response.json()

    # Fetch README
    readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    readme_response = requests.get(readme_url, headers=HEADERS)
    readme_content = ""
    if readme_response.status_code == 200:
        readme_data = readme_response.json()
        readme_content = base64.b64decode(readme_data["content"]).decode("utf-8")

    # Fetch file structure
    contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    contents_response = requests.get(contents_url, headers=HEADERS)
    file_structure = ""
    if contents_response.status_code == 200:
        contents_data = contents_response.json()
        for item in contents_data:
            file_structure += f"📜 `{item['name']}`\n" if item["type"] == "file" else f"📁 **{item['name']}**\n"

    return repo_data, readme_content, file_structure

# New function to fetch code analytics data
def fetch_code_analytics(repo_url):
    """Fetches code analytics data from a GitHub repository."""
    if not GITHUB_TOKEN:
        return None, "❌ Missing GitHub API Token!"

    # Extract owner and repo name from URL
    parts = repo_url.strip("/").split("/")
    if len(parts) < 2:
        return None, "❌ Invalid GitHub repository URL!"

    owner, repo = parts[-2], parts[-1]
    
    # Initialize analytics data
    analytics = {
        "languages": {},
        "commits": [],
        "contributors": [],
        "code_complexity": {}
    }
    
    # Fetch languages
    languages_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    languages_response = requests.get(languages_url, headers=HEADERS)
    if languages_response.status_code == 200:
        analytics["languages"] = languages_response.json()
    
    # Fetch commit history (last 100 commits)
    commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=100"
    commits_response = requests.get(commits_url, headers=HEADERS)
    if commits_response.status_code == 200:
        commits_data = commits_response.json()
        for commit in commits_data:
            try:
                commit_date = datetime.strptime(commit["commit"]["author"]["date"], "%Y-%m-%dT%H:%M:%SZ")
                analytics["commits"].append({
                    "date": commit_date,
                    "author": commit["commit"]["author"]["name"],
                    "message": commit["commit"]["message"]
                })
            except (KeyError, ValueError):
                continue
    
    # Fetch contributors
    contributors_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    contributors_response = requests.get(contributors_url, headers=HEADERS)
    if contributors_response.status_code == 200:
        analytics["contributors"] = contributors_response.json()
    
    # Estimate code complexity by analyzing file types and sizes
    # This is a simplified approach without actual code analysis
    files_data = []
    queue = [(f"https://api.github.com/repos/{owner}/{repo}/contents", "")]
    
    # Limit recursive fetching to avoid API rate limits
    depth_limit = 2
    current_depth = 0
    
    while queue and current_depth < depth_limit:
        current_depth += 1
        level_size = len(queue)
        
        for _ in range(level_size):
            url, path = queue.pop(0)
            response = requests.get(url, headers=HEADERS)
            
            if response.status_code != 200:
                continue
                
            items = response.json()
            if not isinstance(items, list):
                continue
                
            for item in items:
                if item["type"] == "file":
                    ext = os.path.splitext(item["name"])[1].lower()
                    size = item["size"]
                    files_data.append({
                        "name": item["name"],
                        "path": path + "/" + item["name"],
                        "size": size,
                        "extension": ext
                    })
                elif item["type"] == "dir" and current_depth < depth_limit:
                    queue.append((item["url"], path + "/" + item["name"]))
    
    # Calculate complexity scores (using file size as a proxy)
    extensions = [os.path.splitext(f["name"])[1].lower() for f in files_data if f["size"] > 0]
    extension_counts = Counter(extensions)
    
    for ext, count in extension_counts.items():
        if ext:
            # Filter out non-code files
            if ext in ['.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.go', '.ts', '.html', '.css', '.jsx', '.tsx']:
                # Find average file size for this extension
                ext_files = [f for f in files_data if os.path.splitext(f["name"])[1].lower() == ext]
                if ext_files:
                    avg_size = sum(f["size"] for f in ext_files) / len(ext_files)
                    # Complexity score is a function of file count and average size
                    # This is a simplified heuristic, not an actual code complexity metric
                    complexity_score = math.log10(1 + count * avg_size / 1000)
                    analytics["code_complexity"][ext] = {
                        "file_count": count,
                        "avg_size": avg_size,
                        "complexity_score": round(complexity_score, 2)
                    }
    
    return analytics, None

# Generate AI response using OpenAI API
def chat_with_repo(repo_name, repo_description, readme, file_structure, user_query):
    """Chats with OpenAI about the repository."""
    system_prompt = f"""
You are an AI Coding Assistant. The user has provided a GitHub repository with the following details:

- **Repository Name**: {repo_name}
- **Description**: {repo_description}
- **README**: {readme[:500]}... (truncated)
- **File Structure**:
{file_structure}

User Query:
{user_query}

Please answer in Markdown format. If modification is requested, suggest a filename using:
**Suggested Filename:** `filename.ext`
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}]
    )

    response_text = response.choices[0].message.content
    extracted_code = extract_code(response_text)
    suggested_filename = extract_filename(response_text)

    return response_text, extracted_code, suggested_filename

# Function to compare repositories
def compare_repositories(repo1_data, repo1_readme, repo1_files, repo2_data, repo2_readme, repo2_files):
    """Generates a comparison between two repositories using AI."""
    
    system_prompt = f"""
You are an AI Coding Assistant. The user has provided two GitHub repositories to compare:

REPOSITORY 1:
- **Name**: {repo1_data['name']}
- **Description**: {repo1_data['description'] or 'No description'}
- **Stars**: {repo1_data['stargazers_count']} | **Forks**: {repo1_data['forks_count']}
- **README**: {repo1_readme[:300]}... (truncated)
- **File Structure**:
{repo1_files[:500]}... (truncated)

REPOSITORY 2:
- **Name**: {repo2_data['name']}
- **Description**: {repo2_data['description'] or 'No description'}
- **Stars**: {repo2_data['stargazers_count']} | **Forks**: {repo2_data['forks_count']}
- **README**: {repo2_readme[:300]}... (truncated)
- **File Structure**:
{repo2_files[:500]}... (truncated)

Please provide a detailed comparison of these repositories in Markdown format, analyzing:
1. Purpose and functionality
2. Technology stack and dependencies
3. Code organization and structure
4. Community engagement (stars, forks, contributors)
5. Documentation quality
6. Key similarities and differences
7. Strengths and weaknesses of each
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}]
    )

    comparison_text = response.choices[0].message.content
    return comparison_text

# Function to visualize code analytics data
def display_analytics_dashboard(analytics):
    """Creates and displays analytics dashboard for repository."""
    st.subheader("📊 Code Analytics Dashboard")
    
    # Create tabs for different analytics views
    tabs = st.tabs(["Languages", "Commit Activity", "Contributors", "Code Complexity"])
    
    # Languages tab
    with tabs[0]:
        if analytics["languages"]:
            st.subheader("Lines of Code by Language")
            
            # Prepare data for visualization
            langs = list(analytics["languages"].keys())
            sizes = list(analytics["languages"].values())
            
            # Create pie chart
            fig = px.pie(
                names=langs,
                values=sizes,
                title="Repository Language Distribution",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Create bar chart
            fig = px.bar(
                x=langs,
                y=sizes,
                title="Lines of Code by Language",
                labels={"x": "Language", "y": "Bytes of Code"},
                color=langs
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data
            st.subheader("Language Data")
            lang_df = pd.DataFrame({
                "Language": langs,
                "Bytes": sizes,
                "Percentage": [size/sum(sizes)*100 for size in sizes]
            })
            st.dataframe(lang_df)
        else:
            st.info("No language data available.")
    
    # Commit Activity tab
    with tabs[1]:
        if analytics["commits"]:
            st.subheader("Commit Frequency")
            
            # Prepare data for visualization
            dates = [commit["date"] for commit in analytics["commits"]]
            
            # Group commits by date
            date_counts = Counter([d.date() for d in dates])
            
            # Create a full date range with zeros for missing dates
            if len(dates) > 1:
                min_date = min(dates).date()
                max_date = max(dates).date()
                date_range = [(min_date + timedelta(days=i)) for i in range((max_date - min_date).days + 1)]
                
                # Fill missing dates with zero counts
                for date in date_range:
                    if date not in date_counts:
                        date_counts[date] = 0
            
            # Sort dates
            sorted_dates = sorted(date_counts.keys())
            
            # Create line chart
            commit_df = pd.DataFrame({
                "Date": sorted_dates,
                "Commits": [date_counts[date] for date in sorted_dates]
            })
            
            fig = px.line(
                commit_df,
                x="Date",
                y="Commits",
                title="Commit Activity Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show commit authors
            authors = [commit["author"] for commit in analytics["commits"]]
            author_counts = Counter(authors)
            
            # Create bar chart for authors
            author_df = pd.DataFrame({
                "Author": list(author_counts.keys()),
                "Commits": list(author_counts.values())
            }).sort_values("Commits", ascending=False)
            
            fig = px.bar(
                author_df,
                x="Author",
                y="Commits",
                title="Commits by Author",
                color="Author"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No commit data available.")
    
    # Contributors tab
    with tabs[2]:
        if analytics["contributors"]:
            st.subheader("Contributors Statistics")
            
            # Prepare data for visualization
            contrib_df = pd.DataFrame([
                {
                    "Username": contrib["login"],
                    "Contributions": contrib["contributions"],
                    "Profile": contrib["html_url"]
                }
                for contrib in analytics["contributors"]
            ]).sort_values("Contributions", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                contrib_df,
                x="Username",
                y="Contributions",
                title="Contributions by User",
                color="Username"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data
            st.dataframe(contrib_df)
            
            # Calculate contribution distribution
            total_contribs = sum(contrib_df["Contributions"])
            top_contrib_pct = contrib_df["Contributions"].iloc[0] / total_contribs * 100 if not contrib_df.empty else 0
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Contributors", len(contrib_df))
            with col2:
                st.metric("Total Contributions", total_contribs)
            with col3:
                st.metric("Top Contributor %", f"{top_contrib_pct:.1f}%")
        else:
            st.info("No contributor data available.")
    
    # Code Complexity tab
    with tabs[3]:
        if analytics["code_complexity"]:
            st.subheader("Code Complexity Estimates")
            
            # Prepare data for visualization
            complexity_data = []
            for ext, data in analytics["code_complexity"].items():
                complexity_data.append({
                    "Extension": ext,
                    "File Count": data["file_count"],
                    "Avg Size (KB)": data["avg_size"] / 1024,
                    "Complexity Score": data["complexity_score"]
                })
            
            complexity_df = pd.DataFrame(complexity_data).sort_values("Complexity Score", ascending=False)
            
            # Create bar chart for complexity scores
            fig = px.bar(
                complexity_df,
                x="Extension",
                y="Complexity Score",
                title="Code Complexity by File Type",
                color="Extension"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Create scatter plot for file count vs size
            fig = px.scatter(
                complexity_df,
                x="File Count",
                y="Avg Size (KB)",
                size="Complexity Score",
                color="Extension",
                title="File Count vs. Average Size",
                labels={"File Count": "Number of Files", "Avg Size (KB)": "Average File Size (KB)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data
            st.dataframe(complexity_df)
            
            # Note about complexity calculation
            st.info("Note: Complexity scores are simplified estimates based on file sizes and counts, not actual code analysis.")
        else:
            st.info("No code complexity data available.")

# Streamlit UI
st.set_page_config(page_title="GitHub Repo Chatbot", layout="wide")
st.title("🤖 Git Connect By Syndicators")
st.markdown("Enter a GitHub repository URL and chat with the AI to explain, modify, or analyze the code.")

# Add mode selection
mode = st.radio("Select Mode:", ["Single Repository", "Repository Comparison"])

if mode == "Single Repository":
    # Original single repository mode
    repo_url = st.text_input("🔗 GitHub Repository URL:", "")

    if repo_url:
        st.write("🔄 Fetching repository details...")

        repo_data, readme_content, file_structure = fetch_github_repo(repo_url)

        if repo_data:
            st.success(f"✅ Fetched {repo_data['name']}")
            st.write(f"**📌 Description:** {repo_data['description'] or 'No description available'}")
            st.write(f"**⭐ Stars:** {repo_data['stargazers_count']} | **🍴 Forks:** {repo_data['forks_count']}")
            st.write(f"🔗 [View on GitHub]({repo_data['html_url']})")

            # Display README
            if readme_content:
                st.subheader("📖 README")
                st.code(readme_content[:1000] + "...", language="markdown")  # Truncated

            # Display file structure
            if file_structure:
                st.subheader("📂 Repository Files")
                st.markdown(file_structure)
            
            # Fetch and display code analytics
            with st.expander("🔍 View Code Analytics Dashboard", expanded=False):
                analytics_data, error = fetch_code_analytics(repo_url)
                if analytics_data:
                    display_analytics_dashboard(analytics_data)
                else:
                    st.error(f"❌ Failed to fetch analytics data: {error}")

            # Chat UI
            user_query = st.text_area("💡 Ask AI about this repository...")
            if st.button("🔍 Get AI Response"):
                if user_query:
                    with st.spinner("Thinking..."):
                        response_text, extracted_code, suggested_filename = chat_with_repo(
                            repo_data["name"], repo_data["description"], readme_content, file_structure, user_query
                        )

                        st.subheader("🤖 AI Response")
                        st.markdown(response_text)

                        if extracted_code:
                            st.markdown(f"**📂 Suggested Filename:** `{suggested_filename}`")
                            st.code(extracted_code, language="python")

                else:
                    st.warning("⚠️ Please enter a query to proceed.")
        else:
            st.error("❌ Failed to fetch repository details. Please check the URL or API Key.")

else:
    # Repository comparison mode
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Repository 1")
        repo1_url = st.text_input("🔗 GitHub Repository URL 1:", key="repo1")
    
    with col2:
        st.subheader("Repository 2")
        repo2_url = st.text_input("🔗 GitHub Repository URL 2:", key="repo2")
    
    # Only proceed if both URLs are provided
    if repo1_url and repo2_url:
        st.write("🔄 Fetching repository details...")
        
        # Fetch details for both repositories
        repo1_data, repo1_readme, repo1_files = fetch_github_repo(repo1_url)
        repo2_data, repo2_readme, repo2_files = fetch_github_repo(repo2_url)
        
        # Check if both fetches were successful
        if repo1_data and repo2_data:
            # Display repositories side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"✅ Repository 1: {repo1_data['name']}")
                st.write(f"**📌 Description:** {repo1_data['description'] or 'No description available'}")
                st.write(f"**⭐ Stars:** {repo1_data['stargazers_count']} | **🍴 Forks:** {repo1_data['forks_count']}")
                st.write(f"**👥 Contributors:** {repo1_data.get('subscribers_count', 'N/A')}")
                st.write(f"**🔍 Created:** {repo1_data.get('created_at', 'N/A')}")
                st.write(f"**🔄 Last Updated:** {repo1_data.get('updated_at', 'N/A')}")
                
                # Show file structure preview
                st.subheader("📂 File Structure")
                st.markdown(repo1_files[:500] + ("..." if len(repo1_files) > 500 else ""))
                
                # Fetch and display code analytics for repo 1
                analytics1_data, error1 = fetch_code_analytics(repo1_url)
                if analytics1_data:
                    with st.expander("🔍 View Code Analytics Dashboard", expanded=False):
                        display_analytics_dashboard(analytics1_data)
                else:
                    st.error(f"❌ Failed to fetch analytics data: {error1}")
            
            with col2:
                st.success(f"✅ Repository 2: {repo2_data['name']}")
                st.write(f"**📌 Description:** {repo2_data['description'] or 'No description available'}")
                st.write(f"**⭐ Stars:** {repo2_data['stargazers_count']} | **🍴 Forks:** {repo2_data['forks_count']}")
                st.write(f"**👥 Contributors:** {repo2_data.get('subscribers_count', 'N/A')}")
                st.write(f"**🔍 Created:** {repo2_data.get('created_at', 'N/A')}")
                st.write(f"**🔄 Last Updated:** {repo2_data.get('updated_at', 'N/A')}")
                
                # Show file structure preview
                st.subheader("📂 File Structure")
                st.markdown(repo2_files[:500] + ("..." if len(repo2_files) > 500 else ""))
                
                # Fetch and display code analytics for repo 2
                analytics2_data, error2 = fetch_code_analytics(repo2_url)
                if analytics2_data:
                    with st.expander("🔍 View Code Analytics Dashboard", expanded=False):
                        display_analytics_dashboard(analytics2_data)
                else:
                    st.error(f"❌ Failed to fetch analytics data: {error2}")
            
            # Show combined analytics comparison
            if analytics1_data and analytics2_data:
                st.subheader("📊 Analytics Comparison")
                
                # Compare languages
                if analytics1_data["languages"] and analytics2_data["languages"]:
                    st.write("**Language Distribution Comparison**")
                    
                    # Create dataframe with both repos
                    all_langs = set(list(analytics1_data["languages"].keys()) + list(analytics2_data["languages"].keys()))
                    lang_comparison = []
                    
                    for lang in all_langs:
                        bytes1 = analytics1_data["languages"].get(lang, 0)
                        bytes2 = analytics2_data["languages"].get(lang, 0)
                        total1 = sum(analytics1_data["languages"].values())
                        total2 = sum(analytics2_data["languages"].values())
                        
                        lang_comparison.append({
                            "Language": lang,
                            f"{repo1_data['name']} Bytes": bytes1,
                            f"{repo1_data['name']} %": round(bytes1/total1*100 if total1 > 0 else 0, 2),
                            f"{repo2_data['name']} Bytes": bytes2,
                            f"{repo2_data['name']} %": round(bytes2/total2*100 if total2 > 0 else 0, 2),
                        })
                    
                    lang_df = pd.DataFrame(lang_comparison).sort_values(f"{repo1_data['name']} Bytes", ascending=False)
                    st.dataframe(lang_df)
                    
                    # Create bar chart comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=lang_df["Language"],
                        y=lang_df[f"{repo1_data['name']} %"],
                        name=repo1_data['name'],
                        marker_color='rgb(55, 83, 109)'
                    ))
                    fig.add_trace(go.Bar(
                        x=lang_df["Language"],
                        y=lang_df[f"{repo2_data['name']} %"],
                        name=repo2_data['name'],
                        marker_color='rgb(26, 118, 255)'
                    ))
                    
                    fig.update_layout(
                        title='Language Distribution Comparison (%)',
                        xaxis_tickfont_size=14,
                        yaxis=dict(
                            title='Percentage (%)',
                            titlefont_size=16,
                            tickfont_size=14,
                        ),
                        barmode='group',
                        bargap=0.15,
                        bargroupgap=0.1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Compare contributor stats
                if analytics1_data["contributors"] and analytics2_data["contributors"]:
                    st.write("**Contributor Comparison**")
                    
                    # Create metrics comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{repo1_data['name']} Contributors", len(analytics1_data["contributors"]))
                        st.metric(f"{repo1_data['name']} Total Contributions", 
                                  sum(c["contributions"] for c in analytics1_data["contributors"]))
                    with col2:
                        st.metric(f"{repo2_data['name']} Contributors", len(analytics2_data["contributors"]))
                        st.metric(f"{repo2_data['name']} Total Contributions", 
                                  sum(c["contributions"] for c in analytics2_data["contributors"]))
                
                # Compare commit activity
                if analytics1_data["commits"] and analytics2_data["commits"]:
                    st.write("**Commit Activity Comparison**")
                    
                    # Prepare data for visualization
                    dates1 = [commit["date"] for commit in analytics1_data["commits"]]
                    dates2 = [commit["date"] for commit in analytics2_data["commits"]]
                    
                    # Group commits by date
                    date_counts1 = Counter([d.date() for d in dates1])
                    date_counts2 = Counter([d.date() for d in dates2])
                    
                    # Create full date range
                    all_dates = set([d.date() for d in dates1 + dates2])
                    
                    # Create dataframe for comparison
                    commit_comparison = []
                    for date in sorted(all_dates):
                        commit_comparison.append({
                            "Date": date,
                            f"{repo1_data['name']} Commits": date_counts1.get(date, 0),
                            f"{repo2_data['name']} Commits": date_counts2.get(date, 0)
                        })
                    
                    commit_df = pd.DataFrame(commit_comparison)
                    
                    # Create line chart comparison
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=commit_df["Date"],
                        y=commit_df[f"{repo1_data['name']} Commits"],
                        mode='lines+markers',
                        name=repo1_data['name']
                    ))
                    fig.add_trace(go.Scatter(
                        x=commit_df["Date"],
                        y=commit_df[f"{repo2_data['name']} Commits"],
                        mode='lines+markers',
                        name=repo2_data['name']
                    ))
                    
                    fig.update_layout(
                        title='Commit Activity Comparison',
                        xaxis_title='Date',
                        yaxis_title='Number of Commits'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show basic comparison metrics
            st.subheader("📊 Quick Comparison Metrics")
            
            # Create metrics dataframe
            metrics = {
                "Metric": ["Stars", "Forks", "Issue Count", "Language"],
                "Repository 1": [
                    repo1_data["stargazers_count"],
                    repo1_data["forks_count"],
                    repo1_data.get("open_issues_count", "N/A"),
                    repo1_data.get("language", "N/A")
                ],
                "Repository 2": [
                    repo2_data["stargazers_count"],
                    repo2_data["forks_count"],
                    repo2_data.get("open_issues_count", "N/A"),
                    repo2_data.get("language", "N/A")
                ]
            }
            
            metrics_df = pd.DataFrame(metrics)
            st.table(metrics_df)
            
            # AI Comparison button
            if st.button("🧠 Generate AI Comparison Analysis"):
                with st.spinner("Generating detailed comparison..."):
                    comparison_text = compare_repositories(
                        repo1_data, repo1_readme, repo1_files,
                        repo2_data, repo2_readme, repo2_files
                    )
                    st.subheader("🤖 AI Repository Comparison")
                    st.markdown(comparison_text)
            
            # Allow specific feature comparison
            st.subheader("🔍 Compare Specific Features")
            feature_to_compare = st.selectbox(
                "Select feature to compare in detail:",
                ["READMEs", "License", "Programming Languages", "Dependencies"]
            )
            
            if st.button("Compare Selected Feature"):
                with st.spinner(f"Comparing {feature_to_compare}..."):
                    if feature_to_compare == "READMEs":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"{repo1_data['name']} README")
                            st.markdown(repo1_readme[:2000] + ("..." if len(repo1_readme) > 2000 else ""))
                        with col2:
                            st.subheader(f"{repo2_data['name']} README")
                            st.markdown(repo2_readme[:2000] + ("..." if len(repo2_readme) > 2000 else ""))
                    else:
                        # For other features, we'd need additional API calls
                        # This is a placeholder for other comparison types
                        st.info(f"Feature comparison for {feature_to_compare} would require additional GitHub API calls. Implementation pending.")
        
        elif not repo1_data:
            st.error("❌ Failed to fetch Repository 1. Please check the URL.")
        elif not repo2_data:
            st.error("❌ Failed to fetch Repository 2. Please check the URL.")

st.info("🔹 Enter a valid GitHub repository URL above to analyze and chat with it, or choose comparison mode to analyze two repositories side by side.")