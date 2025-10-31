import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from typing import Dict, List, Any, Optional
from datetime import datetime
import io
import os
import tempfile
import numpy as np


class DriftReportGenerator:
    """
    PDF report generator for model drift analysis.
    Creates comprehensive reports with visualizations and statistical summaries.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        sns.set_style("whitegrid")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))
    
    def _create_drift_timeline_chart(self, 
                                    timestamps: List[datetime], 
                                    psi_scores: List[float],
                                    ks_statistics: List[float]) -> str:
        """Create timeline chart of drift metrics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(timestamps, psi_scores, marker='o', linewidth=2, color='#e74c3c')
        ax1.axhline(y=0.25, color='orange', linestyle='--', label='Warning Threshold')
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Critical Threshold')
        ax1.set_ylabel('PSI Score', fontsize=12)
        ax1.set_title('Population Stability Index Over Time', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(timestamps, ks_statistics, marker='s', linewidth=2, color='#3498db')
        ax2.axhline(y=0.2, color='orange', linestyle='--', label='Warning Threshold')
        ax2.axhline(y=0.3, color='red', linestyle='--', label='Critical Threshold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('KS Statistic', fontsize=12)
        ax2.set_title('Kolmogorov-Smirnov Statistic Over Time', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_file.name
    
    def _create_sharpe_chart(self, 
                           timestamps: List[datetime], 
                           sharpe_ratios: List[float]) -> str:
        """Create Sharpe ratio deterioration chart."""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(timestamps, sharpe_ratios, marker='o', linewidth=2, color='#27ae60')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=-0.5, color='red', linestyle='--', label='Critical Threshold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_title('Rolling Sharpe Ratio Performance', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_file.name
    
    def _create_forecast_chart(self, 
                              current_date: datetime,
                              bayesian_days: float,
                              neural_sde_days: float) -> str:
        """Create time-to-failure forecast visualization."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        methods = ['Bayesian\nRegression', 'Neural SDE']
        days = [bayesian_days if bayesian_days > 0 else 0, 
               neural_sde_days if neural_sde_days > 0 else 0]
        colors_bar = ['#3498db', '#e74c3c']
        
        bars = ax.barh(methods, days, color=colors_bar, alpha=0.7)
        
        for i, (bar, day) in enumerate(zip(bars, days)):
            if day > 0:
                ax.text(day, i, f'  {day:.1f} days', va='center', fontsize=12)
            else:
                ax.text(0, i, '  No failure predicted', va='center', fontsize=10)
        
        ax.set_xlabel('Days to Predicted Failure', fontsize=12)
        ax.set_title('Time-to-Failure Forecasts', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_file.name
    
    def generate_pdf_report(self, 
                           model_name: str,
                           drift_analyses: List[Dict[str, Any]],
                           output_path: str) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            model_name: Name of the model
            drift_analyses: List of drift analysis results
            output_path: Path to save the PDF
            
        Returns:
            Path to generated PDF file
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        title = Paragraph(f"Quantora Drift Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        subtitle = Paragraph(f"<b>Model:</b> {model_name}", self.styles['Normal'])
        story.append(subtitle)
        
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_para = Paragraph(f"<b>Generated:</b> {date_str}", self.styles['Normal'])
        story.append(date_para)
        story.append(Spacer(1, 0.3*inch))
        
        if drift_analyses:
            latest = drift_analyses[-1]
            
            summary_header = Paragraph("Executive Summary", self.styles['SectionHeader'])
            story.append(summary_header)
            
            summary_data = [
                ['Metric', 'Value', 'Status'],
                ['PSI Score', f"{latest.get('psi_score', 0):.4f}", 
                 self._get_status_text(latest.get('psi_score', 0), 0.25)],
                ['KS Statistic', f"{latest.get('ks_statistic', 0):.4f}",
                 self._get_status_text(latest.get('ks_statistic', 0), 0.2)],
                ['Drift Severity', latest.get('drift_severity', 'N/A'), ''],
                ['Time to Failure (Bayesian)', 
                 f"{latest.get('time_to_failure_bayesian', -1):.1f} days" if latest.get('time_to_failure_bayesian', -1) > 0 else 'N/A', ''],
                ['Time to Failure (Neural SDE)', 
                 f"{latest.get('time_to_failure_neural_sde', -1):.1f} days" if latest.get('time_to_failure_neural_sde', -1) > 0 else 'N/A', ''],
                ['Failure Probability', f"{latest.get('failure_probability', 0):.2%}", '']
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))
            
            if len(drift_analyses) > 1:
                timestamps = [a['analysis_timestamp'] for a in drift_analyses]
                psi_scores = [a.get('psi_score', 0) for a in drift_analyses]
                ks_stats = [a.get('ks_statistic', 0) for a in drift_analyses]
                
                chart_header = Paragraph("Drift Metrics Timeline", self.styles['SectionHeader'])
                story.append(chart_header)
                
                chart_path = self._create_drift_timeline_chart(timestamps, psi_scores, ks_stats)
                img = Image(chart_path, width=6.5*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
                
                os.unlink(chart_path)
                
                sharpe_ratios = [a.get('rolling_sharpe', 0) for a in drift_analyses]
                if any(s != 0 for s in sharpe_ratios):
                    sharpe_header = Paragraph("Performance Metrics", self.styles['SectionHeader'])
                    story.append(sharpe_header)
                    
                    sharpe_path = self._create_sharpe_chart(timestamps, sharpe_ratios)
                    img_sharpe = Image(sharpe_path, width=6.5*inch, height=2.5*inch)
                    story.append(img_sharpe)
                    story.append(Spacer(1, 0.2*inch))
                    
                    os.unlink(sharpe_path)
                
                forecast_header = Paragraph("Failure Forecasts", self.styles['SectionHeader'])
                story.append(forecast_header)
                
                forecast_path = self._create_forecast_chart(
                    datetime.now(),
                    latest.get('time_to_failure_bayesian', -1),
                    latest.get('time_to_failure_neural_sde', -1)
                )
                img_forecast = Image(forecast_path, width=5*inch, height=3*inch)
                story.append(img_forecast)
                
                os.unlink(forecast_path)
        
        recommendations_header = Paragraph("Recommendations", self.styles['SectionHeader'])
        story.append(Spacer(1, 0.3*inch))
        story.append(recommendations_header)
        
        recommendations = self._generate_recommendations(drift_analyses[-1] if drift_analyses else {})
        for rec in recommendations:
            bullet = Paragraph(f"• {rec}", self.styles['Normal'])
            story.append(bullet)
            story.append(Spacer(1, 0.1*inch))
        
        doc.build(story)
        
        return output_path
    
    def _get_status_text(self, value: float, threshold: float) -> str:
        """Get status text based on threshold."""
        if value < threshold:
            return '✓ OK'
        elif value < threshold * 2:
            return '⚠ Warning'
        else:
            return '✗ Critical'
    
    def _generate_recommendations(self, latest_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        psi = latest_analysis.get('psi_score', 0)
        if psi > 0.5:
            recommendations.append("CRITICAL: Severe feature drift detected. Consider retraining the model immediately.")
        elif psi > 0.25:
            recommendations.append("WARNING: Moderate drift detected. Monitor closely and prepare for retraining.")
        
        ks_stat = latest_analysis.get('ks_statistic', 0)
        if ks_stat > 0.3:
            recommendations.append("Distribution shift is significant. Investigate data pipeline changes.")
        
        ttf_bayes = latest_analysis.get('time_to_failure_bayesian', -1)
        if 0 < ttf_bayes < 14:
            recommendations.append(f"Model failure predicted within {ttf_bayes:.0f} days. Immediate action required.")
        
        failure_prob = latest_analysis.get('failure_probability', 0)
        if failure_prob > 0.7:
            recommendations.append(f"High failure probability ({failure_prob:.0%}). Implement contingency plans.")
        
        if not recommendations:
            recommendations.append("Model performance is stable. Continue regular monitoring.")
        
        return recommendations
