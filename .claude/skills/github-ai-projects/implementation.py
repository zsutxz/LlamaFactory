"""
GitHub AI热门项目监控技能实现
这个模块提供了搜索、分析和报告GitHub AI项目的主要功能
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import re

class GitHubAIAgentMonitor:
    """GitHub AI项目监控器"""
    
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Project-Monitor"
        }
        # AI相关的搜索关键词
        self.ai_keywords = [
            "artificial intelligence",
            "machine learning", 
            "deep learning",
            "neural network",
            "pytorch",
            "tensorflow",
            "scikit-learn",
            "huggingface",
            "langchain",
            "transformer",
            "llm",
            "computer vision",
            "nlp",
            "generative ai"
        ]
    
    def search_ai_projects(self, sort_by="stars", per_page=20) -> List[Dict]:
        """搜索AI相关项目"""
        # 构建搜索查询
        query = " OR ".join(self.ai_keywords)
        query += f" stars:>100 language:python"  # 至少100个星标，Python语言
        
        params = {
            "q": query,
            "sort": sort_by,
            "order": "desc",
            "per_page": per_page
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/search/repositories",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json().get("items", [])
        except Exception as e:
            print(f"搜索项目时出错: {e}")
            return []
    
    def get_repo_details(self, repo_full_name: str) -> Dict:
        """获取仓库详细信息"""
        try:
            response = requests.get(
                f"{self.base_url}/repos/{repo_full_name}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取仓库详情时出错: {e}")
            return {}
    
    def get_repo_commits(self, repo_full_name: str, since_date: str = None) -> List[Dict]:
        """获取仓库提交历史"""
        params = {}
        if since_date:
            params["since"] = since_date
            
        try:
            response = requests.get(
                f"{self.base_url}/repos/{repo_full_name}/commits",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取提交历史时出错: {e}")
            return []
    
    def calculate_activity_score(self, repo: Dict, timeframe_days: int = 7) -> float:
        """计算项目活跃度评分"""
        score = 0.0
        
        # 基础分数（星标数）
        stars = repo.get("stargazers_count", 0)
        if stars > 10000:
            score += 3.0
        elif stars > 1000:
            score += 2.0
        elif stars > 100:
            score += 1.0
        
        # Fork数
        forks = repo.get("forks_count", 0)
        if forks > 1000:
            score += 2.0
        elif forks > 100:
            score += 1.0
        
        # 最近更新时间
        updated_at = repo.get("updated_at", "")
        if updated_at:
            last_update = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            days_since_update = (datetime.now(last_update.tzinfo) - last_update).days
            
            if days_since_update <= 1:
                score += 3.0
            elif days_since_update <= 7:
                score += 2.0
            elif days_since_update <= 30:
                score += 1.0
        
        # 开源协议（有加分）
        if repo.get("license"):
            score += 0.5
        
        # 有README（有加分）
        if repo.get("has_issues", False):
            score += 0.5
        
        return min(10.0, score)
    
    def generate_project_summary(self, repo: Dict, activity_score: float) -> Dict:
        """生成项目简述"""
        name = repo.get("name", "")
        description = repo.get("description", "")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        language = repo.get("language", "")
        updated_at = repo.get("updated_at", "")
        
        # 提取主要技术标签
        topics = repo.get("topics", [])
        tech_tags = [tag for tag in topics if any(keyword in tag.lower() 
                    for keyword in ["pytorch", "tensorflow", "ml", "ai", "deep", "nlp", "cv"])]
        
        # 生成热度分析
        heat_analysis = self._generate_heat_analysis(repo, activity_score)
        
        return {
            "name": name,
            "stars": stars,
            "forks": forks,
            "language": language,
            "updated_at": updated_at,
            "description": description[:100] + "..." if len(description) > 100 else description,
            "tech_tags": tech_tags[:5],  # 最多显示5个标签
            "heat_analysis": heat_analysis,
            "activity_score": round(activity_score, 1),
            "url": repo.get("html_url", "")
        }
    
    def _generate_heat_analysis(self, repo: Dict, score: float) -> str:
        """生成热度分析文本"""
        stars = repo.get("stargazers_count", 0)
        language = repo.get("language", "")
        topics = repo.get("topics", [])
        
        analyses = []
        
        if stars > 10000:
            analyses.append("超高人气项目，社区活跃")
        elif stars > 1000:
            analyses.append("高人气项目，值得关注")
        
        if "machine-learning" in topics or "deep-learning" in topics:
            analyses.append("机器学习核心技术")
        
        if "pytorch" in topics or "tensorflow" in topics:
            analyses.append("主流深度学习框架")
        
        if language == "Python":
            analyses.append("Python生态重要项目")
        
        if score >= 7.0:
            analyses.append("近期非常活跃")
        elif score >= 5.0:
            analyses.append("保持稳定更新")
        
        return " | ".join(analyses) if analyses else "新兴AI项目"
    
    def analyze_timeframe_activity(self, projects: List[Dict], timeframe: str) -> List[Dict]:
        """分析特定时间范围内的项目活跃度"""
        now = datetime.now()
        
        if timeframe == "1day":
            since_date = now - timedelta(days=1)
            time_desc = "最近1天"
        elif timeframe == "1week":
            since_date = now - timedelta(days=7)
            time_desc = "本周"
        elif timeframe == "1month":
            since_date = now - timedelta(days=30)
            time_desc = "最近一个月"
        else:
            return []
        
        active_projects = []
        for project in projects[:10]:  # 分析前10个项目
            full_name = project.get("full_name", "")
            if not full_name:
                continue
                
            # 获取提交历史
            commits = self.get_repo_commits(full_name, since_date.isoformat())
            
            if commits:
                activity_info = {
                    "name": project.get("name", ""),
                    "commits_count": len(commits),
                    "timeframe": time_desc,
                    "last_commit": commits[0].get("commit", {}).get("author", {}).get("date", "") if commits else "",
                    "activity_score": min(10, len(commits) / 2)  # 简化的活跃度评分
                }
                active_projects.append(activity_info)
        
        # 按提交数排序
        return sorted(active_projects, key=lambda x: x["commits_count"], reverse=True)
    
    def generate_report(self, limit: int = 10) -> Dict:
        """生成完整的监控报告"""
        # 搜索热门项目
        projects = self.search_ai_projects(per_page=limit)
        
        # 生成项目详细信息
        detailed_projects = []
        for project in projects:
            activity_score = self.calculate_activity_score(project)
            summary = self.generate_project_summary(project, activity_score)
            detailed_projects.append(summary)
        
        # 按活跃度排序
        detailed_projects.sort(key=lambda x: x["activity_score"], reverse=True)
        
        # 分析不同时间范围的活跃度
        daily_active = self.analyze_timeframe_activity(projects, "1day")
        weekly_active = self.analyze_timeframe_activity(projects, "1week")
        monthly_active = self.analyze_timeframe_activity(projects, "1month")
        
        return {
            "top_projects": detailed_projects[:10],
            "daily_active": daily_active[:5],
            "weekly_active": weekly_active[:5],
            "monthly_active": monthly_active[:5],
            "generated_at": datetime.now().isoformat(),
            "total_analyzed": len(projects)
        }

def format_github_ai_report(report_data: Dict) -> str:
    """格式化GitHub AI项目报告"""
    top_projects = report_data.get("top_projects", [])
    daily_active = report_data.get("daily_active", [])
    weekly_active = report_data.get("weekly_active", [])
    monthly_active = report_data.get("monthly_active", [])
    generated_at = report_data.get("generated_at", "")
    total_analyzed = report_data.get("total_analyzed", 0)
    
    # 格式化生成时间
    if generated_at:
        dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        formatted_time = "未知"
    
    report = f"""
🚀 GitHub AI热门项目监控报告
📅 更新时间：{formatted_time}
🔍 分析项目数：{total_analyzed}

## 📈 热度排行榜（Top 10）

"""
    
    # 添加排行榜项目
    for i, project in enumerate(top_projects, 1):
        medals = ["🥇", "🥈", "🥉"]
        medal = medals[i-1] if i <= 3 else f"#{i}"
        
        tags_str = " | ".join(project["tech_tags"]) if project["tech_tags"] else "无标签"
        
        report += f"""
### {medal} {project["name"]}
⭐ {project["stars"]:,} | 🍴 {project["forks"]:,} | 📅 {project["updated_at"][:10]}
🏷️ {tags_str} | 💻 {project["language"]}
📝 {project["description"]}
🔥 {project["heat_analysis"]}
📊 活跃度评分：{project["activity_score"]}/10.0
🔗 {project["url"]}

"""
    
    # 添加时间维度分析
    report += "## ⏰ 时间维度分析\n\n"
    
    if daily_active:
        report += "### 🔥 最近1天活跃项目\n"
        for project in daily_active:
            report += f"- **{project["name"]}**: {project["commits_count"]} 次提交，活跃度 {project["activity_score"]}/10\n"
        report += "\n"
    
    if weekly_active:
        report += "### 📅 本周热门项目\n"
        for project in weekly_active:
            report += f"- **{project["name"]}**: {project["commits_count"]} 次提交，活跃度 {project["activity_score"]}/10\n"
        report += "\n"
    
    if monthly_active:
        report += "### 📊 最近一个月趋势\n"
        for project in monthly_active:
            report += f"- **{project["name"]}**: {project["commits_count"]} 次提交，活跃度 {project["activity_score"]}/10\n"
        report += "\n"
    
    # 添加重点推荐
    report += "## 🎯 重点推荐\n\n"
    if top_projects:
        top_3 = top_projects[:3]
        for i, project in enumerate(top_3, 1):
            report += f"**{i}. {project["name"]}** - {project["heat_analysis"]}\n"
    
    report += f"""
## 📋 数据洞察
- 本次分析了 {total_analyzed} 个AI相关项目
- 平均活跃度评分：{sum(p["activity_score"] for p in top_projects) / len(top_projects):.1f}/10.0
- 最高星标项目：{top_projects[0]["name"] if top_projects else "无"} ({top_projects[0]["stars"]:,} ⭐)
- 最活跃语言：Python
- 主要技术方向：深度学习、机器学习框架、NLP工具

💡 *建议关注活跃度评分较高的项目，这些项目通常具有更好的社区支持和发展前景。*
"""
    
    return report

# 示例使用代码
if __name__ == "__main__":
    monitor = GitHubAIAgentMonitor()
    report_data = monitor.generate_report(limit=20)
    formatted_report = format_github_ai_report(report_data)
    print(formatted_report)