import matplotlib.pyplot as plt
import numpy as np
import os

def save_individual_plots(predictor, output_dir='plots', dpi=300):
    """
    Save each plot as a separate high-quality figure
    
    Args:
        predictor: The trained RULPredictor object
        output_dir: Directory to save plots
        dpi: Resolution for saved plots
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for publication quality
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    plots_saved = []
    
    try:
        # 1. RMS Values - Channels 1-4
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, color in zip(range(1, 5), colors):
            col = f'ch{i}_rms'
            if col in predictor.df.columns:
                ax.plot(predictor.df[col], alpha=0.8, linewidth=2.5, 
                       label=f'Channel {i}', color=color)
        ax.set_title('RMS Values - Channels 1-4', fontweight='bold', pad=20)
        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('RMS Value', fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_rms_channels_1-4.png', dpi=dpi, bbox_inches='tight')
        plt.savefig(f'{output_dir}/01_rms_channels_1-4.pdf', bbox_inches='tight')
        plt.close()
        plots_saved.append("01_rms_channels_1-4")
        
        # 2. RMS Values - Channels 5-8
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for i, color in zip(range(5, 9), colors):
            col = f'ch{i}_rms'
            if col in predictor.df.columns:
                ax.plot(predictor.df[col], alpha=0.8, linewidth=2.5, 
                       label=f'Channel {i}', color=color)
        ax.set_title('RMS Values - Channels 5-8', fontweight='bold', pad=20)
        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('RMS Value', fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_rms_channels_5-8.png', dpi=dpi, bbox_inches='tight')
        plt.savefig(f'{output_dir}/02_rms_channels_5-8.pdf', bbox_inches='tight')
        plt.close()
        plots_saved.append("02_rms_channels_5-8")
        
        # 3. MWUT State Indicators
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(predictor.mwut_indicators, 'b-', linewidth=3, alpha=0.8)
        ax.fill_between(range(len(predictor.mwut_indicators)), 
                       predictor.mwut_indicators, alpha=0.3)
        ax.set_title('Mann-Whitney U Test State Indicators', fontweight='bold', pad=20)
        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('State Indicator Value', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_mwut_state_indicators.png', dpi=dpi, bbox_inches='tight')
        plt.savefig(f'{output_dir}/03_mwut_state_indicators.pdf', bbox_inches='tight')
        plt.close()
        plots_saved.append("03_mwut_state_indicators")
        
        # 4. Damage Accumulation
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(predictor.damage_accumulation, 'r-', linewidth=3, alpha=0.9, 
               label='Damage Accumulation')
        ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, 
                  label='Failure Threshold (0.8)', alpha=0.8)
        ax.fill_between(range(len(predictor.damage_accumulation)), 
                       predictor.damage_accumulation, alpha=0.2, color='red')
        ax.set_title('Damage Accumulation Over Time', fontweight='bold', pad=20)
        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('Damage Accumulation', fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_damage_accumulation.png', dpi=dpi, bbox_inches='tight')
        plt.savefig(f'{output_dir}/04_damage_accumulation.pdf', bbox_inches='tight')
        plt.close()
        plots_saved.append("04_damage_accumulation")
        
        # 5. Model Performance
        if hasattr(predictor, 'y_pred') and not np.isinf(predictor.rmse):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(predictor.y_test, predictor.y_pred, alpha=0.6, s=50, 
                      edgecolors='black', linewidth=0.5)
            ax.plot([0, 1], [0, 1], 'r--', lw=3, alpha=0.8, label='Perfect Prediction')
            ax.set_xlabel('Actual DA Values', fontweight='bold')
            ax.set_ylabel('Predicted DA Values', fontweight='bold')
            ax.set_title(f'Model Performance\nRMSE: {predictor.rmse:.4f} | R²: {predictor.r2:.4f}', 
                        fontweight='bold', pad=20)
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/05_model_performance.png', dpi=dpi, bbox_inches='tight')
            plt.savefig(f'{output_dir}/05_model_performance.pdf', bbox_inches='tight')
            plt.close()
            plots_saved.append("05_model_performance")
        
        # 6. Training History
        if hasattr(predictor, 'history'):
            fig, ax = plt.subplots(figsize=(12, 8))
            epochs = range(1, len(predictor.history.history['loss']) + 1)
            ax.plot(epochs, predictor.history.history['loss'], 'b-', linewidth=2, 
                   label='Training Loss', alpha=0.8)
            if 'val_loss' in predictor.history.history:
                ax.plot(epochs, predictor.history.history['val_loss'], 'r-', linewidth=2, 
                       label='Validation Loss', alpha=0.8)
            ax.set_title('Model Training History', fontweight='bold', pad=20)
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss', fontweight='bold')
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/06_training_history.png', dpi=dpi, bbox_inches='tight')
            plt.savefig(f'{output_dir}/06_training_history.pdf', bbox_inches='tight')
            plt.close()
            plots_saved.append("06_training_history")
        
        # 7-9. RUL Predictions from different time points
        prediction_points = [100, 200, 300]
        for idx, point in enumerate(prediction_points, 7):
            fig, ax = plt.subplots(figsize=(12, 8))
            try:
                if point < len(predictor.damage_accumulation) - predictor.sequence_length:
                    rul, future_pred = predictor.predict_rul(point)
                    if future_pred is not None and len(future_pred) > 0:
                        time_steps = range(point, point + len(future_pred))
                        ax.plot(time_steps, future_pred, 'b-', linewidth=3, 
                               label='RUL Prediction', alpha=0.8)
                        ax.axhline(y=0.8, color='r', linestyle='--', linewidth=2, 
                                  alpha=0.7, label='Failure Threshold')
                        if rul is not None:
                            ax.axvline(x=point + rul, color='orange', linestyle=':', 
                                      linewidth=3, label=f'Predicted Failure (RUL: {rul} steps)')
                        ax.fill_between(time_steps, future_pred, alpha=0.2, color='blue')
                    else:
                        ax.text(0.5, 0.5, 'Prediction Failed', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=16, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
                else:
                    ax.text(0.5, 0.5, 'Insufficient Data for Prediction', 
                           transform=ax.transAxes, ha='center', va='center', fontsize=16,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            except Exception as e:
                ax.text(0.5, 0.5, f'Prediction Error:\n{str(e)[:50]}...', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
            
            ax.set_title(f'RUL Prediction from Time Step {point}', fontweight='bold', pad=20)
            ax.set_xlabel('Time Step', fontweight='bold')
            ax.set_ylabel('Predicted Damage Accumulation', fontweight='bold')
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{idx:02d}_rul_prediction_step_{point}.png', 
                       dpi=dpi, bbox_inches='tight')
            plt.savefig(f'{output_dir}/{idx:02d}_rul_prediction_step_{point}.pdf', 
                       bbox_inches='tight')
            plt.close()
            plots_saved.append(f"{idx:02d}_rul_prediction_step_{point}")
        
        # 10. Combined Indicators
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(predictor.mwut_indicators, alpha=0.8, linewidth=2, 
               label='MWUT State Indicator', color='blue')
        ax.plot(predictor.damage_accumulation, alpha=0.8, linewidth=2, 
               label='Damage Accumulation', color='red')
        ax.set_title('Combined Health Indicators', fontweight='bold', pad=20)
        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('Normalized Value', fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_combined_indicators.png', dpi=dpi, bbox_inches='tight')
        plt.savefig(f'{output_dir}/10_combined_indicators.pdf', bbox_inches='tight')
        plt.close()
        plots_saved.append("10_combined_indicators")
        
        # 11. Energy Features Evolution (if available)
        energy_cols = [col for col in predictor.df.columns if 'energy' in col.lower()][:4]
        if energy_cols:
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for col, color in zip(energy_cols, colors):
                if col in predictor.df.columns:
                    normalized_data = (predictor.df[col] - predictor.df[col].min()) / \
                                    (predictor.df[col].max() - predictor.df[col].min() + 1e-8)
                    ax.plot(normalized_data, alpha=0.8, linewidth=2, 
                           label=col.replace('_', ' ').title(), color=color)
            ax.set_title('Normalized Energy Features Evolution', fontweight='bold', pad=20)
            ax.set_xlabel('Time Step', fontweight='bold')
            ax.set_ylabel('Normalized Energy Value', fontweight='bold')
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/11_energy_features.png', dpi=dpi, bbox_inches='tight')
            plt.savefig(f'{output_dir}/11_energy_features.pdf', bbox_inches='tight')
            plt.close()
            plots_saved.append("11_energy_features")
        
        # 12. Statistical Features (Skewness and Kurtosis)
        stat_features = []
        for i in range(1, 5):  # First 4 channels
            for stat in ['skew', 'kurtosis']:
                col = f'ch{i}_{stat}'
                if col in predictor.df.columns:
                    stat_features.append(col)
        
        if stat_features:
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(stat_features)))
            for col, color in zip(stat_features[:6], colors):  # Limit to 6 for clarity
                ax.plot(predictor.df[col], alpha=0.7, linewidth=2, 
                       label=col.replace('_', ' ').title(), color=color)
            ax.set_title('Statistical Features Evolution', fontweight='bold', pad=20)
            ax.set_xlabel('Time Step', fontweight='bold')
            ax.set_ylabel('Feature Value', fontweight='bold')
            ax.legend(frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/12_statistical_features.png', dpi=dpi, bbox_inches='tight')
            plt.savefig(f'{output_dir}/12_statistical_features.pdf', bbox_inches='tight')
            plt.close()
            plots_saved.append("12_statistical_features")
        
        # Create a summary report
        with open(f'{output_dir}/plot_summary.txt', 'w') as f:
            f.write("RUL Analysis - Plot Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total plots saved: {len(plots_saved)}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Resolution: {dpi} DPI\n\n")
            f.write("Files saved:\n")
            for plot in plots_saved:
                f.write(f"  - {plot}.png\n")
                f.write(f"  - {plot}.pdf\n")
            
            f.write(f"\nModel Performance Summary:\n")
            if hasattr(predictor, 'rmse'):
                f.write(f"  - RMSE: {predictor.rmse:.4f}\n")
                f.write(f"  - MAE: {predictor.mae:.4f}\n")
                f.write(f"  - R²: {predictor.r2:.4f}\n")
            f.write(f"  - Dataset size: {len(predictor.df)} samples\n")
            f.write(f"  - Features used: {predictor.X_scaled.shape[1]}\n")
            f.write(f"  - MWUT range: [{np.min(predictor.mwut_indicators):.4f}, {np.max(predictor.mwut_indicators):.4f}]\n")
            f.write(f"  - Final DA: {predictor.damage_accumulation[-1]:.4f}\n")
        
        print(f"\n✅ Successfully saved {len(plots_saved)} plots to '{output_dir}/' directory")
        print(f"📊 Each plot saved in both PNG ({dpi} DPI) and PDF formats")
        print(f"📋 Summary report saved as 'plot_summary.txt'")
        
        return plots_saved
        
    except Exception as e:
        print(f"❌ Error saving plots: {e}")
        import traceback
        traceback.print_exc()
        return []

# Updated main function to include individual plot saving
def run_analysis_with_individual_plots():
    """Run analysis and save individual plots"""
    try:
        # Run the original analysis
        predictor = run_robust_analysis()
        
        if predictor is not None:
            print("\n" + "="*60)
            print("SAVING INDIVIDUAL PLOTS")
            print("="*60)
            
            # Save individual plots
            saved_plots = save_individual_plots(predictor, output_dir='rul_analysis_plots', dpi=300)
            
            if saved_plots:
                print(f"\n🎯 All plots successfully saved!")
                print(f"📁 Check the 'rul_analysis_plots' folder for individual files")
            
        return predictor
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

if __name__ == "__main__":
    predictor = run_analysis_with_individual_plots()