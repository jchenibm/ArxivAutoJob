# ArxivAutoJob

This repo automatically collects arxiv papers with [arxiv_mcp_project](https://github.com/blazickjp/arxiv-mcp-server) and generates AI-powered summaries on a weekly basis using GitHub Actions.

## Features

- **Automated Paper Collection**: Weekly collection of AI and Machine Learning papers from arXiv
- **AI-Powered Summaries**: Uses DeepSeek AI to generate concise summaries of academic papers
- **RSS Subscription**: Provides RSS feed for easy tracking of new papers and summaries
- **GitHub Actions Integration**: Fully automated workflow running weekly

## Update
> 2025-04-07:
I don't suppose I should use/rely on AI to read all of them.
But I want to use AI to provide me a quick summary.
As leaving it at (comprehensive-analysis)[https://github.com/blazickjp/arxiv-mcp-server/blob/main/src/arxiv_mcp_server/prompts/deep_research_analysis_prompt.py#L21C2-L21C24] part.

## RSS Subscription

To subscribe to paper updates, add the following URL to your RSS reader:

```
https://jchenibm.github.io/ArxivAutoJob/rss.xml
```

Recommended RSS readers:
- Feedly (web and mobile)
- Inoreader (web and mobile)
- Reeder (macOS/iOS)

## How It Works

1. GitHub Actions triggers weekly (Friday 3:00 AM Beijing Time)
2. Collects papers from arXiv CS.AI and CS.LG categories from the past week
3. Downloads PDFs and converts them to Markdown
4. Uses AI to generate summaries with key information
5. Generates RSS feed with the latest paper summaries
6. Stores results as GitHub Action artifacts
