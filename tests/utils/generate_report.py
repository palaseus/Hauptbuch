#!/usr/bin/env python3
"""
Hauptbuch Test Report Generator
Generates comprehensive HTML test reports with metrics and analysis.
"""

import os
import json
import time
import glob
from datetime import datetime
from typing import Dict, List, Any
import argparse

class TestReportGenerator:
    """Generates comprehensive test reports"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "categories": {},
            "metrics": {},
            "failures": [],
            "recommendations": []
        }
    
    def collect_test_results(self) -> Dict[str, Any]:
        """Collect test results from all categories"""
        categories = ["infrastructure", "integration", "contracts", "api", "performance", "security"]
        
        for category in categories:
            result_file = os.path.join(self.results_dir, f"{category}-tests.txt")
            if os.path.exists(result_file):
                self.report_data["categories"][category] = self.parse_test_file(result_file)
            else:
                self.report_data["categories"][category] = {"status": "not_run", "tests": 0, "passed": 0, "failed": 0}
        
        return self.report_data
    
    def parse_test_file(self, file_path: str) -> Dict[str, Any]:
        """Parse test result file"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Simple parsing - in a real implementation, this would be more sophisticated
        lines = content.split('\n')
        test_count = len([line for line in lines if 'test' in line.lower()])
        passed_count = len([line for line in lines if 'passed' in line.lower() or '✓' in line])
        failed_count = len([line for line in lines if 'failed' in line.lower() or '✗' in line])
        
        return {
            "status": "completed" if failed_count == 0 else "failed",
            "tests": test_count,
            "passed": passed_count,
            "failed": failed_count,
            "content": content
        }
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate test metrics"""
        total_tests = sum(cat.get("tests", 0) for cat in self.report_data["categories"].values())
        total_passed = sum(cat.get("passed", 0) for cat in self.report_data["categories"].values())
        total_failed = sum(cat.get("failed", 0) for cat in self.report_data["categories"].values())
        
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        self.report_data["metrics"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": pass_rate,
            "categories_run": len([cat for cat in self.report_data["categories"].values() if cat.get("status") != "not_run"])
        }
        
        return self.report_data["metrics"]
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        metrics = self.report_data["metrics"]
        
        if metrics["pass_rate"] >= 95:
            status = "excellent"
            color = "green"
        elif metrics["pass_rate"] >= 80:
            status = "good"
            color = "orange"
        else:
            status = "needs_attention"
            color = "red"
        
        self.report_data["summary"] = {
            "status": status,
            "color": color,
            "overall_health": metrics["pass_rate"],
            "total_categories": len(self.report_data["categories"]),
            "categories_passed": len([cat for cat in self.report_data["categories"].values() if cat.get("status") == "completed"])
        }
        
        return self.report_data["summary"]
    
    def identify_failures(self) -> List[Dict[str, Any]]:
        """Identify and categorize failures"""
        failures = []
        
        for category, data in self.report_data["categories"].items():
            if data.get("failed", 0) > 0:
                failures.append({
                    "category": category,
                    "failed_tests": data.get("failed", 0),
                    "status": data.get("status", "unknown"),
                    "priority": "high" if category in ["security", "infrastructure"] else "medium"
                })
        
        self.report_data["failures"] = failures
        return failures
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check pass rate
        if self.report_data["metrics"]["pass_rate"] < 80:
            recommendations.append("Overall pass rate is below 80%. Review failed tests and fix critical issues.")
        
        # Check specific categories
        for category, data in self.report_data["categories"].items():
            if data.get("status") == "failed":
                if category == "security":
                    recommendations.append("Security tests failed. This is critical - review security vulnerabilities immediately.")
                elif category == "infrastructure":
                    recommendations.append("Infrastructure tests failed. Check network setup and deployment scripts.")
                elif category == "contracts":
                    recommendations.append("Smart contract tests failed. Review contract logic and deployment.")
                else:
                    recommendations.append(f"{category.title()} tests failed. Review and fix issues.")
        
        # Check if any categories weren't run
        not_run = [cat for cat, data in self.report_data["categories"].items() if data.get("status") == "not_run"]
        if not_run:
            recommendations.append(f"Some test categories were not run: {', '.join(not_run)}. Consider running all tests.")
        
        self.report_data["recommendations"] = recommendations
        return recommendations
    
    def generate_html_report(self) -> str:
        """Generate HTML test report"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hauptbuch Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .summary {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .status-{self.report_data['summary']['color']} {{
            background: {'#d4edda' if self.report_data['summary']['color'] == 'green' else '#fff3cd' if self.report_data['summary']['color'] == 'orange' else '#f8d7da'} !important;
            border-left-color: {'#28a745' if self.report_data['summary']['color'] == 'green' else '#ffc107' if self.report_data['summary']['color'] == 'orange' else '#dc3545'} !important;
        }}
        .categories {{
            padding: 30px;
        }}
        .category-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .category-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .category-header {{
            background: #f8f9fa;
            padding: 15px;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
        }}
        .category-content {{
            padding: 15px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-completed {{
            background: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .status-not_run {{
            background: #fff3cd;
            color: #856404;
        }}
        .failures {{
            padding: 30px;
            background: #f8f9fa;
        }}
        .failure-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #dc3545;
        }}
        .recommendations {{
            padding: 30px;
            background: #e9ecef;
        }}
        .recommendation-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }}
        .footer {{
            background: #343a40;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hauptbuch Test Report</h1>
            <p>Generated on {self.report_data['timestamp']}</p>
        </div>
        
        <div class="summary">
            <h2>Test Summary</h2>
            <div class="summary-grid">
                <div class="metric-card status-{self.report_data['summary']['color']}">
                    <h3>Overall Health</h3>
                    <div class="value">{self.report_data['metrics']['pass_rate']:.1f}%</div>
                </div>
                <div class="metric-card">
                    <h3>Total Tests</h3>
                    <div class="value">{self.report_data['metrics']['total_tests']}</div>
                </div>
                <div class="metric-card">
                    <h3>Passed</h3>
                    <div class="value">{self.report_data['metrics']['total_passed']}</div>
                </div>
                <div class="metric-card">
                    <h3>Failed</h3>
                    <div class="value">{self.report_data['metrics']['total_failed']}</div>
                </div>
            </div>
        </div>
        
        <div class="categories">
            <h2>Test Categories</h2>
            <div class="category-grid">
"""
        
        # Add category cards
        for category, data in self.report_data["categories"].items():
            status_class = f"status-{data.get('status', 'not_run')}"
            html += f"""
                <div class="category-card">
                    <div class="category-header">
                        {category.title()} Tests
                        <span class="status-badge {status_class}">{data.get('status', 'not_run')}</span>
                    </div>
                    <div class="category-content">
                        <p><strong>Tests:</strong> {data.get('tests', 0)}</p>
                        <p><strong>Passed:</strong> {data.get('passed', 0)}</p>
                        <p><strong>Failed:</strong> {data.get('failed', 0)}</p>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # Add failures section
        if self.report_data["failures"]:
            html += """
        <div class="failures">
            <h2>Failed Tests</h2>
"""
            for failure in self.report_data["failures"]:
                html += f"""
            <div class="failure-item">
                <h4>{failure['category'].title()} Tests</h4>
                <p><strong>Failed:</strong> {failure['failed_tests']} tests</p>
                <p><strong>Priority:</strong> {failure['priority']}</p>
            </div>
"""
            html += """
        </div>
"""
        
        # Add recommendations section
        if self.report_data["recommendations"]:
            html += """
        <div class="recommendations">
            <h2>Recommendations</h2>
"""
            for recommendation in self.report_data["recommendations"]:
                html += f"""
            <div class="recommendation-item">
                <p>{recommendation}</p>
            </div>
"""
            html += """
        </div>
"""
        
        # Add detailed results
        html += """
        <div class="categories">
            <h2>Detailed Results</h2>
"""
        for category, data in self.report_data["categories"].items():
            if data.get("content"):
                html += f"""
            <h3>{category.title()} Test Output</h3>
            <pre>{data['content']}</pre>
"""
        
        html += """
        </div>
        
        <div class="footer">
            <p>Hauptbuch Blockchain Test Suite</p>
            <p>Generated by automated testing framework</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def save_report(self, output_file: str = None) -> str:
        """Save the test report"""
        if not output_file:
            output_file = os.path.join(self.results_dir, "test-report.html")
        
        html_content = self.generate_html_report()
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def save_json_report(self, output_file: str = None) -> str:
        """Save JSON version of the report"""
        if not output_file:
            output_file = os.path.join(self.results_dir, "test-report.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate Hauptbuch test reports")
    parser.add_argument("--results-dir", required=True, help="Directory containing test results")
    parser.add_argument("--output", help="Output file for HTML report")
    parser.add_argument("--json", help="Output file for JSON report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return 1
    
    generator = TestReportGenerator(args.results_dir)
    
    print("Collecting test results...")
    generator.collect_test_results()
    
    print("Calculating metrics...")
    generator.calculate_metrics()
    
    print("Generating summary...")
    generator.generate_summary()
    
    print("Identifying failures...")
    generator.identify_failures()
    
    print("Generating recommendations...")
    generator.generate_recommendations()
    
    print("Generating HTML report...")
    html_file = generator.save_report(args.output)
    print(f"HTML report saved to: {html_file}")
    
    if args.json:
        print("Generating JSON report...")
        json_file = generator.save_json_report(args.json)
        print(f"JSON report saved to: {json_file}")
    
    # Print summary
    summary = generator.report_data["summary"]
    metrics = generator.report_data["metrics"]
    
    print(f"\nTest Summary:")
    print(f"  Overall Health: {metrics['pass_rate']:.1f}%")
    print(f"  Total Tests: {metrics['total_tests']}")
    print(f"  Passed: {metrics['total_passed']}")
    print(f"  Failed: {metrics['total_failed']}")
    print(f"  Status: {summary['status']}")
    
    if generator.report_data["failures"]:
        print(f"\nFailures:")
        for failure in generator.report_data["failures"]:
            print(f"  - {failure['category']}: {failure['failed_tests']} failed")
    
    if generator.report_data["recommendations"]:
        print(f"\nRecommendations:")
        for rec in generator.report_data["recommendations"]:
            print(f"  - {rec}")
    
    return 0

if __name__ == "__main__":
    exit(main())
