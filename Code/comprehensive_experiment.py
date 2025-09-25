import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class EfficientToolWearDetector:
    def __init__(self, sampling_rate=1000000, segment_duration=0.00666, overlap_ratio=0.9):
        """
        Efficient implementation - process subset of data for faster results
        """
        self.sampling_rate = sampling_rate
        self.segment_duration = segment_duration
        self.segment_length = int(segment_duration * sampling_rate)
        self.overlap_ratio = overlap_ratio
        self.step_size = int(self.segment_length * (1 - overlap_ratio))
        
    def load_and_subsample_data(self, directory_path, max_files=5, subsample_factor=10):
        """Load data and take every Nth segment to reduce processing time"""
        files = []
        data = []
        
        file_list = [f for f in sorted(os.listdir(directory_path)) if f.endswith('.mat')]
        
        for i, filename in enumerate(file_list[:max_files]):  # Process only first max_files
            filepath = os.path.join(directory_path, filename)
            try:
                mat_data = loadmat(filepath)
                
                # Extract signal data
                signal_data = None
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        signal_data = mat_data[key].flatten()
                        break
                
                if signal_data is not None and len(signal_data) > self.segment_length:
                    # Subsample the data to reduce processing time
                    subsampled_data = signal_data[::subsample_factor]
                    files.append(filename)
                    data.append(subsampled_data)
                    print(f"Loaded {filename}: {len(subsampled_data)} samples (subsampled)")
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return files, data
    
    def create_segments_efficient(self, signal_data, max_segments=1000):
        """Create segments with limit to control processing time"""
        if len(signal_data) < self.segment_length:
            return np.array([])
        
        # Adjust step size and segment length for subsampled data
        adjusted_segment_length = self.segment_length // 10  # Since we subsampled by 10
        adjusted_step_size = self.step_size // 10
        
        segments = []
        count = 0
        
        for i in range(0, len(signal_data) - adjusted_segment_length + 1, adjusted_step_size):
            if count >= max_segments:  # Limit number of segments
                break
                
            segment = signal_data[i:i + adjusted_segment_length]
            if np.std(segment) > 1e-10:  # Avoid flat segments
                segments.append(segment)
                count += 1
        
        return np.array(segments)
    
    def extract_features_vectorized(self, segments):
        """Vectorized feature extraction for speed"""
        if len(segments) == 0:
            return {}
        
        print(f"Extracting features from {len(segments)} segments using vectorized operations...")
        
        features = {}
        
        # IQR - vectorized
        features['iqr'] = np.percentile(segments, 75, axis=1) - np.percentile(segments, 25, axis=1)
        
        # Peak Count - simplified but fast
        features['peak_count'] = self.peak_count_fast(segments)
        
        # Zero Crossing Rate - vectorized
        features['zcr'] = self.zcr_fast(segments)
        
        # Rank-based Entropy - simplified
        features['rank_based_entropy'] = self.entropy_fast(segments)
        
        # Fractal Geometry - simplified
        features['fractal_geometry_indicator'] = self.fractal_fast(segments)
        
        # Chaos Quantifier - simplified
        features['chaos_quantifier'] = self.chaos_fast(segments)
        
        return features
    
    def peak_count_fast(self, segments):
        """Fast peak counting using vectorized operations"""
        # Simple peak detection: count local maxima
        peak_counts = []
        for segment in segments:
            # Find peaks using simple logic
            diff1 = np.diff(segment)
            diff2 = np.diff(diff1)
            peaks = np.sum(diff2 < -np.std(segment) * 0.1)
            peak_counts.append(peaks)
        return np.array(peak_counts)
    
    def zcr_fast(self, segments):
        """Fast zero crossing rate"""
        # Vectorized zero crossing detection
        segments_centered = segments - np.mean(segments, axis=1, keepdims=True)
        sign_changes = np.diff(np.sign(segments_centered), axis=1)
        zcr = np.count_nonzero(sign_changes, axis=1) / (segments.shape[1] - 1)
        return zcr
    
    def entropy_fast(self, segments, n_bins=20):
        """Fast entropy calculation"""
        entropies = []
        for segment in segments:
            hist, _ = np.histogram(segment, bins=n_bins)
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) > 0:
                p = hist / np.sum(hist)
                entropy = -np.sum(p * np.log2(p + 1e-12))
            else:
                entropy = 0
            entropies.append(entropy)
        return np.array(entropies)
    
    def fractal_fast(self, segments):
        """Simplified fractal dimension"""
        fractal_dims = []
        for segment in segments:
            # Simplified box-counting approach
            try:
                # Calculate variance at different scales
                scales = [1, 2, 4, 8]
                variances = []
                
                for scale in scales:
                    if len(segment) > scale * 4:
                        downsampled = segment[::scale]
                        variances.append(np.var(downsampled))
                
                if len(variances) > 2:
                    # Linear fit in log space
                    log_scales = np.log(scales[:len(variances)])
                    log_vars = np.log(np.array(variances) + 1e-12)
                    slope = np.polyfit(log_scales, log_vars, 1)[0]
                    fractal_dims.append(abs(slope))
                else:
                    fractal_dims.append(1.0)
            except:
                fractal_dims.append(1.0)
        
        return np.array(fractal_dims)
    
    def chaos_fast(self, segments):
        """Simplified chaos quantifier"""
        chaos_values = []
        for segment in segments:
            # Use sample entropy approximation
            try:
                # Calculate successive differences
                diffs = np.diff(segment)
                # Measure of unpredictability
                chaos_measure = np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-12)
                chaos_values.append(chaos_measure)
            except:
                chaos_values.append(0)
        
        return np.array(chaos_values)
    
    def analyze_feature_separation(self, healthy_features, faulty_features):
        """Analyze how well features separate the two classes"""
        separation_results = {}
        
        print("\nFeature Separation Analysis:")
        print("-" * 60)
        print(f"{'Feature':<30} {'Healthy Mean':<12} {'Faulty Mean':<12} {'Separation'}")
        print("-" * 60)
        
        for feature_name in healthy_features.keys():
            healthy_data = healthy_features[feature_name]
            faulty_data = faulty_features[feature_name]
            
            healthy_mean = np.mean(healthy_data)
            faulty_mean = np.mean(faulty_data)
            healthy_std = np.std(healthy_data)
            faulty_std = np.std(faulty_data)
            
            # Calculate separation measure (Cohen's d)
            pooled_std = np.sqrt(((len(healthy_data) - 1) * healthy_std**2 + 
                                 (len(faulty_data) - 1) * faulty_std**2) / 
                                (len(healthy_data) + len(faulty_data) - 2))
            
            cohens_d = abs(healthy_mean - faulty_mean) / (pooled_std + 1e-12)
            
            separation_results[feature_name] = {
                'healthy_mean': healthy_mean,
                'faulty_mean': faulty_mean,
                'separation': cohens_d
            }
            
            print(f"{feature_name:<30} {healthy_mean:<12.4f} {faulty_mean:<12.4f} {cohens_d:.4f}")
        
        return separation_results
    
    def calculate_performance_optimized(self, healthy_features, faulty_features):
        """Optimized performance calculation"""
        results = {}
        
        for feature_name in healthy_features.keys():
            healthy_data = healthy_features[feature_name]
            faulty_data = faulty_features[feature_name]
            
            # Create ground truth
            y_true = np.concatenate([np.zeros(len(healthy_data)), np.ones(len(faulty_data))])
            combined_data = np.concatenate([healthy_data, faulty_data])
            
            # Use optimal threshold based on ROC analysis
            from sklearn.metrics import roc_curve
            try:
                fpr, tpr, thresholds = roc_curve(y_true, combined_data)
                # Find threshold that maximizes (tpr - fpr)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                predictions = (combined_data >= optimal_threshold).astype(int)
                
            except:
                # Fallback to median threshold
                optimal_threshold = np.median(combined_data)
                predictions = (combined_data >= optimal_threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, zero_division=0)
            recall = recall_score(y_true, predictions, zero_division=0)
            
            results[feature_name] = {
                'accuracy': accuracy * 100,
                'precision': precision * 100,
                'recall': recall * 100,
                'threshold': optimal_threshold
            }
        
        return results
    
    def plot_features_comparison(self, healthy_features, faulty_features):
        """Create comparison plots"""
        feature_names = list(healthy_features.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature_name in enumerate(feature_names):
            if i >= 6:
                break
                
            ax = axes[i]
            
            healthy_data = healthy_features[feature_name]
            faulty_data = faulty_features[feature_name]
            
            # Take a sample for plotting (to avoid overcrowded plots)
            sample_size = min(1000, len(healthy_data), len(faulty_data))
            
            healthy_sample = np.random.choice(healthy_data, sample_size, replace=False)
            faulty_sample = np.random.choice(faulty_data, sample_size, replace=False)
            
            # Plot as histograms for better visualization
            ax.hist(healthy_sample, bins=30, alpha=0.7, color='green', label='Healthy', density=True)
            ax.hist(faulty_sample, bins=30, alpha=0.7, color='red', label='Faulty', density=True)
            
            ax.set_title(f'{feature_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
        
        plt.tight_layout()
        plt.suptitle('Feature Distribution Comparison', y=1.02, fontsize=14)
        plt.show()

def main_efficient():
    """Efficient main function that runs quickly"""
    
    print("=== Efficient Tool Wear Detection ===")
    print("This version processes subset of data for fast results\n")
    
    # Initialize detector
    detector = EfficientToolWearDetector()
    
    # Define paths
    healthy_path = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\MCT\N1320"
    faulty_path = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\MCT\T1320"
    
    print("Loading and subsampling data for efficient processing...")
    
    # Load subsampled data (first 3 files, every 10th sample)
    healthy_files, healthy_data = detector.load_and_subsample_data(healthy_path, max_files=3, subsample_factor=10)
    faulty_files, faulty_data = detector.load_and_subsample_data(faulty_path, max_files=3, subsample_factor=10)
    
    print(f"\nProcessing {len(healthy_data)} healthy files and {len(faulty_data)} faulty files...")
    
    # Extract features efficiently
    all_healthy_features = {}
    all_faulty_features = {}
    
    print("\nProcessing healthy data...")
    for i, data in enumerate(healthy_data):
        segments = detector.create_segments_efficient(data, max_segments=500)  # Limit segments
        if len(segments) > 0:
            print(f"  File {i+1}: {len(segments)} segments")
            features = detector.extract_features_vectorized(segments)
            
            for feature_name, feature_values in features.items():
                if feature_name not in all_healthy_features:
                    all_healthy_features[feature_name] = []
                all_healthy_features[feature_name].extend(feature_values)
    
    print("\nProcessing faulty data...")
    for i, data in enumerate(faulty_data):
        segments = detector.create_segments_efficient(data, max_segments=500)  # Limit segments
        if len(segments) > 0:
            print(f"  File {i+1}: {len(segments)} segments")
            features = detector.extract_features_vectorized(segments)
            
            for feature_name, feature_values in features.items():
                if feature_name not in all_faulty_features:
                    all_faulty_features[feature_name] = []
                all_faulty_features[feature_name].extend(feature_values)
    
    # Convert to numpy arrays
    for feature_name in all_healthy_features.keys():
        all_healthy_features[feature_name] = np.array(all_healthy_features[feature_name])
        all_faulty_features[feature_name] = np.array(all_faulty_features[feature_name])
    
    # Analyze feature separation
    separation_results = detector.analyze_feature_separation(all_healthy_features, all_faulty_features)
    
    # Calculate performance
    performance = detector.calculate_performance_optimized(all_healthy_features, all_faulty_features)
    
    # Display results
    print("\nPerformance Results:")
    print("-" * 70)
    print(f"{'Feature':<30} {'Accuracy (%)':<12} {'Precision (%)':<13} {'Recall (%)'}")
    print("-" * 70)
    
    total_accuracy = 0
    feature_count = 0
    
    for feature_name, metrics in performance.items():
        print(f"{feature_name:<30} {metrics['accuracy']:<12.1f} {metrics['precision']:<13.1f} {metrics['recall']:.1f}")
        total_accuracy += metrics['accuracy']
        feature_count += 1
    
    avg_accuracy = total_accuracy / feature_count
    print("-" * 70)
    print(f"{'Average':<30} {avg_accuracy:<12.1f}")
    print("-" * 70)
    
    # Create plots
    detector.plot_features_comparison(all_healthy_features, all_faulty_features)
    
    # Recommendations
    print(f"\nResults Summary:")
    print(f"- Average accuracy: {avg_accuracy:.1f}%")
    
    if avg_accuracy > 80:
        print("- Excellent performance! Ready for full dataset processing.")
        print("- Features show good separation between healthy and faulty conditions.")
    elif avg_accuracy > 60:
        print("- Good performance. Consider fine-tuning parameters.")
        print("- Some features show promise for tool wear detection.")
    else:
        print("- Performance needs improvement. Consider:")
        print("  * Adjusting feature extraction parameters")
        print("  * Different segmentation approach")
        print("  * Alternative feature implementations")
    
    # Identify best features
    best_features = sorted(performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    print(f"\nBest performing features:")
    for i, (feature_name, metrics) in enumerate(best_features[:3]):
        print(f"{i+1}. {feature_name}: {metrics['accuracy']:.1f}% accuracy")
    
    return detector, all_healthy_features, all_faulty_features, performance

if __name__ == "__main__":
    # Run efficient version
    detector, healthy_features, faulty_features, performance = main_efficient()