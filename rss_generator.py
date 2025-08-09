"""RSS generator for ArxivAutoJob."""

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from download import get_paper_path


class ArxivRSSGenerator:
    """Generate RSS feed for arXiv papers."""
    
    def __init__(self, output_file: str = "rss.xml"):
        self.output_file = output_file
        self.feed_title = "ArxivAutoJob - AI/ML 论文更新"
        self.feed_description = "每周自动更新的AI和机器学习领域最新论文摘要"
        self.feed_link = "https://github.com/jchenibm/ArxivAutoJob"
        
    def parse_summary_file(self, summary_file: str = "summary.md") -> List[Dict[str, Any]]:
        """Parse the summary.md file and extract paper information."""
        papers = []
        
        if not os.path.exists(summary_file):
            print(f"Summary file {summary_file} not found")
            return papers
            
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by paper ID header
        paper_sections = content.split('# ')
        
        for section in paper_sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if len(lines) < 3:
                continue
                
            # Extract paper ID from first line
            paper_id = lines[0].strip()
            if not paper_id or '/' not in paper_id:
                continue
                
            # Extract content and PDF link
            content_lines = lines[1:-1]  # Exclude first (ID) and last (PDF link) lines
            pdf_link = lines[-1] if lines[-1].startswith('https://arxiv.org/pdf/') else ""
            
            # Try to extract title from markdown file if available
            title = paper_id  # Default to paper ID
            paper_markdown_file = get_paper_path(paper_id)
            if os.path.exists(paper_markdown_file):
                try:
                    with open(paper_markdown_file, 'r', encoding='utf-8') as f:
                        # First line is usually the title in markdown format
                        first_line = f.readline().strip()
                        if first_line.startswith('# '):
                            title = first_line[2:]  # Remove '# ' prefix
                except Exception as e:
                    print(f"Error reading markdown file for {paper_id}: {e}")
            
            paper_info = {
                'id': paper_id,
                'title': title,
                'content': '\n'.join(content_lines),
                'pdf_link': pdf_link,
                'pub_date': datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            }
            
            papers.append(paper_info)
            
        return papers
    
    def generate_rss(self, papers: List[Dict[str, Any]]) -> str:
        """Generate RSS XML from papers list."""
        # Create root element
        rss = ET.Element('rss', version='2.0')
        
        # Create channel element
        channel = ET.SubElement(rss, 'channel')
        
        # Add channel metadata
        title = ET.SubElement(channel, 'title')
        title.text = self.feed_title
        
        description = ET.SubElement(channel, 'description')
        description.text = self.feed_description
        
        link = ET.SubElement(channel, 'link')
        link.text = self.feed_link
        
        last_build_date = ET.SubElement(channel, 'lastBuildDate')
        last_build_date.text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        
        # Add papers as items
        for paper in papers:
            item = ET.SubElement(channel, 'item')
            
            item_title = ET.SubElement(item, 'title')
            item_title.text = paper['title']
            
            item_description = ET.SubElement(item, 'description')
            item_description.text = paper['content']
            
            item_link = ET.SubElement(item, 'link')
            item_link.text = paper['pdf_link'] if paper['pdf_link'] else f"https://arxiv.org/abs/{paper['id']}"
            
            item_guid = ET.SubElement(item, 'guid')
            item_guid.text = paper['id']
            
            item_pub_date = ET.SubElement(item, 'pubDate')
            item_pub_date.text = paper['pub_date']
        
        # Pretty print the XML
        rough_string = ET.tostring(rss, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def save_rss(self, rss_content: str) -> None:
        """Save RSS content to file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(rss_content)
        print(f"RSS feed saved to {self.output_file}")
    
    async def generate_feed(self) -> None:
        """Main method to generate RSS feed."""
        print("Generating RSS feed...")
        
        # Parse summary file
        papers = self.parse_summary_file()
        print(f"Found {len(papers)} papers")
        
        # Generate RSS
        rss_content = self.generate_rss(papers)
        
        # Save RSS
        self.save_rss(rss_content)
        
        print("RSS feed generation completed")


# For testing purposes
if __name__ == "__main__":
    generator = ArxivRSSGenerator()
    asyncio.run(generator.generate_feed())