#!/usr/bin/env python
"""
Create Tourism-Based Visualizations
Generate bar graphs and charts showing Nepal tourism patterns
"""
import os
import sys
import django
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place

class TourismVisualizer:
    def __init__(self, csv_file='nepal_tourism_crowd_data.csv'):
        self.csv_file = csv_file
        self.df = None
        self.load_data()
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load the tourism data"""
        if os.path.exists(self.csv_file):
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.df)} rows of tourism data")
        else:
            print(f"CSV file {self.csv_file} not found!")
            return False
        return True
    
    def create_monthly_tourism_patterns(self):
        """Create bar graph showing monthly tourism patterns"""
        plt.figure(figsize=(14, 8))
        
        # Monthly average crowd levels
        monthly_avg = self.df.groupby('month')['crowdlevel'].mean().reset_index()
        
        # Create bar plot
        bars = plt.bar(monthly_avg['month'], monthly_avg['crowdlevel'], 
                      color='skyblue', edgecolor='navy', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Customize the plot
        plt.title('Nepal Tourism Patterns by Month (Crowd Levels)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Average Crowd Level (%)', fontsize=12, fontweight='bold')
        
        # Set x-axis labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(range(1, 13), month_names, rotation=45)
        
        # Add season annotations
        seasons = ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer',
                  'Summer', 'Summer', 'Autumn', 'Autumn', 'Autumn', 'Winter']
        
        for i, (month, season) in enumerate(zip(range(1, 13), seasons)):
            plt.text(i+1, monthly_avg.iloc[i]['crowdlevel'] + 5, 
                    season, ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='darkred')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('monthly_tourism_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Monthly tourism patterns chart saved as 'monthly_tourism_patterns.png'")
    
    def create_category_comparison(self):
        """Create bar graph comparing crowd levels by category"""
        plt.figure(figsize=(16, 10))
        
        # Category average crowd levels
        category_avg = self.df.groupby('category')['crowdlevel'].mean().sort_values(ascending=False)
        
        # Create bar plot
        bars = plt.bar(range(len(category_avg)), category_avg.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(category_avg))),
                      edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, category_avg.values)):
            plt.text(bar.get_x() + bar.get_width()/2., value + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Customize the plot
        plt.title('Average Crowd Levels by Place Category', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Place Categories', fontsize=12, fontweight='bold')
        plt.ylabel('Average Crowd Level (%)', fontsize=12, fontweight='bold')
        
        # Set x-axis labels
        plt.xticks(range(len(category_avg)), category_avg.index, rotation=45, ha='right')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('category_crowd_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Category comparison chart saved as 'category_crowd_comparison.png'")
    
    def create_tourist_season_analysis(self):
        """Create bar graph showing tourist season patterns"""
        plt.figure(figsize=(12, 8))
        
        # Tourist season analysis
        season_avg = self.df.groupby('tourist_season')['crowdlevel'].mean().reindex(['Low', 'Shoulder', 'Peak'])
        
        # Create bar plot
        colors = ['lightcoral', 'gold', 'lightgreen']
        bars = plt.bar(season_avg.index, season_avg.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, season_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2., value + 2,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Customize the plot
        plt.title('Crowd Levels by Tourist Season', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Tourist Season', fontsize=12, fontweight='bold')
        plt.ylabel('Average Crowd Level (%)', fontsize=12, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('tourist_season_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Tourist season analysis chart saved as 'tourist_season_analysis.png'")
    
    def create_festival_impact(self):
        """Create bar graph showing festival impact on crowd levels"""
        plt.figure(figsize=(10, 8))
        
        # Festival period analysis
        festival_avg = self.df.groupby('festival_period')['crowdlevel'].mean()
        
        # Create bar plot
        colors = ['lightblue', 'orange']
        bars = plt.bar(festival_avg.index, festival_avg.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, festival_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2., value + 2,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Customize the plot
        plt.title('Festival Impact on Crowd Levels', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Festival Period', fontsize=12, fontweight='bold')
        plt.ylabel('Average Crowd Level (%)', fontsize=12, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('festival_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Festival impact analysis chart saved as 'festival_impact_analysis.png'")
    
    def create_time_slot_analysis(self):
        """Create bar graph showing crowd levels by time slot"""
        plt.figure(figsize=(10, 8))
        
        # Time slot analysis
        time_avg = self.df.groupby('time_slot')['crowdlevel'].mean().reindex(['morning', 'afternoon', 'evening'])
        
        # Create bar plot
        colors = ['lightyellow', 'lightcoral', 'lightblue']
        bars = plt.bar(time_avg.index, time_avg.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, time_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2., value + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Customize the plot
        plt.title('Crowd Levels by Time of Day', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Time Slot', fontsize=12, fontweight='bold')
        plt.ylabel('Average Crowd Level (%)', fontsize=12, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('time_slot_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Time slot analysis chart saved as 'time_slot_analysis.png'")
    
    def create_district_comparison(self):
        """Create bar graph comparing crowd levels by district"""
        plt.figure(figsize=(12, 8))
        
        # District average crowd levels
        district_avg = self.df.groupby('district')['crowdlevel'].mean().sort_values(ascending=False)
        
        # Create bar plot
        colors = ['lightgreen', 'lightblue', 'lightcoral']
        bars = plt.bar(district_avg.index, district_avg.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, district_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2., value + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Customize the plot
        plt.title('Average Crowd Levels by District', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('District', fontsize=12, fontweight='bold')
        plt.ylabel('Average Crowd Level (%)', fontsize=12, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('district_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ District comparison chart saved as 'district_comparison.png'")
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with multiple charts"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Nepal Tourism Crowd Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Monthly patterns
        monthly_avg = self.df.groupby('month')['crowdlevel'].mean()
        axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Monthly Tourism Patterns')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Crowd Level (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Category comparison
        category_avg = self.df.groupby('category')['crowdlevel'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(category_avg)), category_avg.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(category_avg))))
        axes[0, 1].set_title('Crowd Levels by Category')
        axes[0, 1].set_xlabel('Categories')
        axes[0, 1].set_ylabel('Crowd Level (%)')
        axes[0, 1].set_xticks(range(len(category_avg)))
        axes[0, 1].set_xticklabels(category_avg.index, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Tourist season
        season_avg = self.df.groupby('tourist_season')['crowdlevel'].mean().reindex(['Low', 'Shoulder', 'Peak'])
        axes[0, 2].bar(season_avg.index, season_avg.values, 
                      color=['lightcoral', 'gold', 'lightgreen'])
        axes[0, 2].set_title('Tourist Season Analysis')
        axes[0, 2].set_xlabel('Season')
        axes[0, 2].set_ylabel('Crowd Level (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Time slots
        time_avg = self.df.groupby('time_slot')['crowdlevel'].mean().reindex(['morning', 'afternoon', 'evening'])
        axes[1, 0].bar(time_avg.index, time_avg.values, 
                      color=['lightyellow', 'lightcoral', 'lightblue'])
        axes[1, 0].set_title('Time Slot Analysis')
        axes[1, 0].set_xlabel('Time Slot')
        axes[1, 0].set_ylabel('Crowd Level (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Festival impact
        festival_avg = self.df.groupby('festival_period')['crowdlevel'].mean()
        axes[1, 1].bar(festival_avg.index, festival_avg.values, 
                      color=['lightblue', 'orange'])
        axes[1, 1].set_title('Festival Impact')
        axes[1, 1].set_xlabel('Festival Period')
        axes[1, 1].set_ylabel('Crowd Level (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. District comparison
        district_avg = self.df.groupby('district')['crowdlevel'].mean().sort_values(ascending=False)
        axes[1, 2].bar(district_avg.index, district_avg.values, 
                      color=['lightgreen', 'lightblue', 'lightcoral'])
        axes[1, 2].set_title('District Comparison')
        axes[1, 2].set_xlabel('District')
        axes[1, 2].set_ylabel('Crowd Level (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tourism_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive dashboard saved as 'tourism_analysis_dashboard.png'")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("🎨 Generating Nepal Tourism Visualizations...\n")
        
        if not self.load_data():
            return
        
        # Create individual charts
        self.create_monthly_tourism_patterns()
        self.create_category_comparison()
        self.create_tourist_season_analysis()
        self.create_festival_impact()
        self.create_time_slot_analysis()
        self.create_district_comparison()
        
        # Create comprehensive dashboard
        self.create_comprehensive_dashboard()
        
        print("\n🎉 All visualizations completed!")
        print("📊 Generated charts:")
        print("   • monthly_tourism_patterns.png")
        print("   • category_crowd_comparison.png")
        print("   • tourist_season_analysis.png")
        print("   • festival_impact_analysis.png")
        print("   • time_slot_analysis.png")
        print("   • district_comparison.png")
        print("   • tourism_analysis_dashboard.png")

if __name__ == "__main__":
    visualizer = TourismVisualizer()
    visualizer.generate_all_visualizations() 