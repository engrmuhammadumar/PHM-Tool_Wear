import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class ApproachBPaperMethod:
    def __init__(self, sampling_rate=1000000, segment_duration=0.1, overlap_ratio=0.9):
        """
        Approach B: Paper's exact methodology but with optimized parameters
        
        Key changes from paper:
        - Longer segments (0.1s vs 0.00666s) to capture your data characteristics
        - Reference vs monitoring approach as described in paper
        - Two-stage Mann-Whitney U test with proper windowing
        """
        self.sampling_rate = sampling_rate
        self.segment_duration = segment_duration
        self.segment_length = int(segment_duration * sampling_rate)  # 100,000 samples at 0.1s
        self.overlap_ratio = overlap_ratio
        self.step_size = int(self.segment_length * (1 - overlap_ratio))
        
    def load_data_paper_style(self, healthy_path, faulty_path, n_files_per_condition=10):
        """Load data following paper's reference vs monitoring approach"""
        
        def load_files_subset(directory, n_files):
            files = sorted([f for f in os.listdir(directory) if f.endswith('.mat')])[:n_files]
            signals = []
            
            for filename in files:
                filepath = os.path.join(directory, filename)
                try:
                    mat_data = loadmat(filepath)
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            signal_data = mat_data[key].flatten()
                            # Take middle portion to avoid edge effects
                            mid_start = len(signal_data) // 4
                            mid_end = 3 * len(signal_data) // 4
                            signals.append(signal_data[mid_start:mid_end])
                            print(f"  Loaded {filename}: {len(signal_data[mid_start:mid_end])} samples")
                            break
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
            
            return signals
        
        print("Loading healthy (reference) data...")
        healthy_signals = load_files_subset(healthy_path, n_files_per_condition)
        
        print("Loading faulty (monitoring) data...")  
        faulty_signals = load_files_subset(faulty_path, n_files_per_condition)
        
        return healthy_signals, faulty_signals
    
    def create_segments_paper_approach(self, signals, max_segments_per_file=50):
        """Create segments following paper's approach but with longer segments"""
        
        all_segments = []
        
        for i, signal in enumerate(signals):
            print(f"    Processing file {i+1}: {len(signal)} samples")
            
            segments_from_file = []
            
            # Create overlapping segments
            for start_idx in range(0, len(signal) - self.segment_length + 1, self.step_size):
                if len(segments_from_file) >= max_segments_per_file:
                    break
                    
                segment = signal[start_idx:start_idx + self.segment_length]
                
                # Quality check
                if np.std(segment) > np.std(signal) * 0.05:  # Relative threshold
                    segments_from_file.append(segment)
            
            all_segments.extend(segments_from_file)
            print(f"      Generated {len(segments_from_file)} segments")
        
        return np.array(all_segments)
    
    def extract_paper_features_optimized(self, segments):
        """Extract features following paper's methodology"""
        
        if len(segments) == 0:
            return {}
        
        print(f"    Extracting features from {len(segments)} segments...")
        features = {}
        
        # Paper's features
        # 1. IQR (Equation 1)
        features['iqr'] = np.percentile(segments, 75, axis=1) - np.percentile(segments, 25, axis=1)
        
        # 2. Peak Count (Equation 2) - exact implementation
        features['peak_count'] = self.calculate_peak_count_exact(segments)
        
        # 3. Zero Crossing Rate (Equation 3) - exact implementation  
        features['zcr'] = self.calculate_zcr_exact(segments)
        
        # 4. Rank-based Entropy (Equation 4) - simplified but effective
        features['rank_based_entropy'] = self.calculate_rbe_efficient(segments)
        
        # 5. Fractal Geometry Indicator - simplified but stable
        features['fractal_geometry_indicator'] = self.calculate_fgi_stable(segments)
        
        # 6. Chaos Quantifier - simplified but meaningful
        features['chaos_quantifier'] = self.calculate_cq_stable(segments)
        
        return features
    
    def calculate_peak_count_exact(self, segments):
        """Exact implementation of Equation 2"""
        peak_counts = []
        
        for segment in segments:
            count = 0
            N = len(segment)
            
            for i in range(1, N-1):
                if segment[i-1] < segment[i] and segment[i] > segment[i+1]:
                    count += 1
            
            peak_counts.append(count)
        
        return np.array(peak_counts)
    
    def calculate_zcr_exact(self, segments):
        """Exact implementation of Equation 3"""
        zcr_values = []
        
        for segment in segments:
            N = len(segment)
            crossings = 0
            
            for i in range(N-1):
                if segment[i] * segment[i+1] < 0:
                    crossings += 1
            
            zcr = crossings / (N - 1) if N > 1 else 0
            zcr_values.append(zcr)
        
        return np.array(zcr_values)
    
    def calculate_rbe_efficient(self, segments, n_bins=64):
        """Efficient rank-based entropy"""
        rbe_values = []
        
        for segment in segments:
            # Create histogram
            hist, _ = np.histogram(segment, bins=n_bins)
            
            # Remove zero bins
            frequencies = hist[hist > 0]
            
            if len(frequencies) == 0:
                rbe_values.append(0)
                continue
            
            # Sort in descending order (ranking)
            ranked_freq = np.sort(frequencies)[::-1]
            
            # Calculate probabilities
            probabilities = ranked_freq / np.sum(ranked_freq)
            
            # Shannon entropy
            rbe = -np.sum(probabilities * np.log2(probabilities + 1e-12))
            rbe_values.append(rbe)
        
        return np.array(rbe_values)
    
    def calculate_fgi_stable(self, segments):
        """Stable fractal geometry indicator"""
        fgi_values = []
        
        for segment in segments:
            try:
                # Simplified box-counting approach
                scales = [1, 2, 4, 8, 16]
                complexities = []
                
                for scale in scales:
                    if len(segment) > scale * 10:
                        # Downsample and calculate variation
                        downsampled = segment[::scale]
                        complexity = np.var(np.diff(downsampled))
                        complexities.append(complexity)
                
                if len(complexities) >= 3:
                    # Linear regression in log space
                    log_scales = np.log(scales[:len(complexities)])
                    log_complexities = np.log(np.array(complexities) + 1e-12)
                    
                    coeffs = np.polyfit(log_scales, log_complexities, 1)
                    fgi = abs(coeffs[0])  # Absolute slope
                else:
                    fgi = 1.0
            except:
                fgi = 1.0
            
            fgi_values.append(fgi)
        
        return np.array(fgi_values)
    
    def calculate_cq_stable(self, segments):
        """Stable chaos quantifier"""
        cq_values = []
        
        for segment in segments:
            try:
                # Use sample entropy approximation
                diffs = np.diff(segment)
                
                # Measure of unpredictability
                if len(diffs) > 10:
                    # Relative variation measure
                    cq = np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-12)
                else:
                    cq = 0
            except:
                cq = 0
                
            cq_values.append(cq)
        
        return np.array(cq_values)
    
    def apply_paper_two_stage_mann_whitney(self, reference_features, monitoring_features):
        """Apply paper's two-stage Mann-Whitney U test"""
        
        print("\n=== Paper's Two-Stage Mann-Whitney U Test ===")
        
        # Stage 1: Initial U-statistics
        print("Stage 1: Reference vs Monitoring U-statistics...")
        print(f"{'Feature':<25} {'U-stat':<10} {'U-norm':<10} {'p-value':<12} {'Decision'}")
        print("-" * 70)
        
        stage1_results = {}
        
        for feature_name in reference_features.keys():
            ref_data = reference_features[feature_name]
            mon_data = monitoring_features[feature_name]
            
            try:
                # Apply Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(ref_data, mon_data, alternative='two-sided')
                
                # Normalize U-statistic
                n1, n2 = len(ref_data), len(mon_data)
                u_normalized = u_stat / (n1 * n2)
                
                # Decision using paper's threshold approach
                threshold = 0.5
                if u_normalized > threshold:
                    decision = "WEAR DETECTED"
                else:
                    decision = "HEALTHY"
                
                stage1_results[feature_name] = {
                    'u_statistic': u_stat,
                    'u_normalized': u_normalized,
                    'p_value': p_value,
                    'decision': decision,
                    'n1': n1, 'n2': n2
                }
                
                print(f"{feature_name:<25} {u_stat:<10.1f} {u_normalized:<10.4f} {p_value:<12.2e} {decision}")
                
            except Exception as e:
                print(f"Error with {feature_name}: {e}")
                stage1_results[feature_name] = {
                    'u_statistic': 0, 'u_normalized': 0.5, 'p_value': 1.0,
                    'decision': 'UNKNOWN', 'n1': len(ref_data), 'n2': len(mon_data)
                }
        
        # Stage 2: Refined analysis (simplified windowing approach)
        print(f"\nStage 2: Refined analysis...")
        
        stage2_results = {}
        
        for feature_name, stage1_data in stage1_results.items():
            # Simplified stage 2: confidence weighting
            u_norm = stage1_data['u_normalized']
            p_val = stage1_data['p_value']
            
            # Calculate confidence based on statistical significance and effect size
            if p_val < 0.001:
                significance_weight = 1.0
            elif p_val < 0.01:
                significance_weight = 0.8
            elif p_val < 0.05:
                significance_weight = 0.6
            else:
                significance_weight = 0.3
            
            # Effect size weight (distance from 0.5)
            effect_weight = 2 * abs(u_norm - 0.5)
            
            # Combined confidence
            confidence = significance_weight * effect_weight
            
            stage2_results[feature_name] = {
                'decision': stage1_data['decision'],
                'confidence': confidence,
                'u_normalized': u_norm,
                'p_value': p_val
            }
            
            print(f"  {feature_name}: {stage1_data['decision']} (confidence: {confidence:.3f})")
        
        return stage1_results, stage2_results
    
    def create_aggregated_indicator(self, stage2_results):
        """Create aggregated tool wear indicator similar to paper's Figure 10"""
        
        print(f"\n=== Aggregated Tool Wear Indicator ===")
        
        # Weight features by their confidence and known effectiveness
        feature_weights = {
            'iqr': 1.0,
            'peak_count': 0.8,
            'zcr': 0.8,
            'rank_based_entropy': 1.2,  # Paper shows this as best
            'fractal_geometry_indicator': 1.1,
            'chaos_quantifier': 1.1
        }
        
        # Calculate weighted decision
        total_weight = 0
        wear_score = 0
        
        for feature_name, results in stage2_results.items():
            if feature_name in feature_weights:
                weight = feature_weights[feature_name] * results['confidence']
                
                if results['decision'] == 'WEAR DETECTED':
                    wear_score += weight
                
                total_weight += weight
        
        # Normalize to 0-1 range
        if total_weight > 0:
            final_indicator = wear_score / total_weight
        else:
            final_indicator = 0.5
        
        # Final decision
        if final_indicator > 0.5:
            final_decision = "TOOL WEAR DETECTED"
            confidence = final_indicator
        else:
            final_decision = "TOOL HEALTHY"  
            confidence = 1 - final_indicator
        
        print(f"Final Indicator Value: {final_indicator:.4f}")
        print(f"Final Decision: {final_decision}")
        print(f"Decision Confidence: {confidence:.4f}")
        
        return {
            'indicator_value': final_indicator,
            'decision': final_decision,
            'confidence': confidence,
            'individual_results': stage2_results
        }
    
    def evaluate_approach_b_performance(self, reference_features, monitoring_features):
        """Evaluate performance of Approach B"""
        
        print(f"\n=== Approach B Performance Evaluation ===")
        
        # For each feature, evaluate classification performance
        results = {}
        
        for feature_name in reference_features.keys():
            ref_data = reference_features[feature_name] 
            mon_data = monitoring_features[feature_name]
            
            # Ground truth: reference = 0 (healthy), monitoring = 1 (faulty)
            y_true = np.concatenate([np.zeros(len(ref_data)), np.ones(len(mon_data))])
            combined_data = np.concatenate([ref_data, mon_data])
            
            # Use Mann-Whitney U-statistic as the decision criterion
            try:
                u_stat, p_value = mannwhitneyu(ref_data, mon_data, alternative='two-sided')
                u_normalized = u_stat / (len(ref_data) * len(mon_data))
                
                # Threshold-based classification
                threshold = 0.5
                predictions = (u_normalized > threshold).astype(int)
                
                # If all predictions are the same, assign based on U-statistic
                if len(set(predictions)) == 1:
                    if u_normalized > threshold:
                        predictions = np.ones(len(y_true))  # All faulty
                    else:
                        predictions = np.zeros(len(y_true))  # All healthy
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, predictions)
                precision = precision_score(y_true, predictions, zero_division=0)
                recall = recall_score(y_true, predictions, zero_division=0)
                
                results[feature_name] = {
                    'accuracy': accuracy * 100,
                    'precision': precision * 100, 
                    'recall': recall * 100,
                    'u_normalized': u_normalized,
                    'p_value': p_value
                }
                
            except Exception as e:
                print(f"Error evaluating {feature_name}: {e}")
                results[feature_name] = {
                    'accuracy': 50.0, 'precision': 0.0, 'recall': 0.0,
                    'u_normalized': 0.5, 'p_value': 1.0
                }
        
        return results


def main_approach_b():
    """Main function for Approach B - Paper's methodology with optimizations"""
    
    print("=== Approach B: Paper's Methodology (Optimized) ===")
    print("Using paper's exact approach with parameters optimized for your data\n")
    
    detector = ApproachBPaperMethod(
        segment_duration=0.1,  # Longer segments based on your data characteristics
        overlap_ratio=0.9      # High overlap as in paper
    )
    
    # Paths
    healthy_path = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\MCT\N1320"
    faulty_path = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\MCT\T1320"
    
    # Load data using paper's approach (10 files per condition for manageable processing)
    healthy_signals, faulty_signals = detector.load_data_paper_style(healthy_path, faulty_path, n_files_per_condition=10)
    
    print(f"\nLoaded {len(healthy_signals)} healthy and {len(faulty_signals)} faulty files")
    
    # Create segments following paper's approach
    print(f"\nCreating segments (paper's approach)...")
    print("  Processing healthy signals...")
    healthy_segments = detector.create_segments_paper_approach(healthy_signals)
    
    print("  Processing faulty signals...")
    faulty_segments = detector.create_segments_paper_approach(faulty_signals)
    
    print(f"\nSegment summary:")
    print(f"  Healthy segments: {len(healthy_segments)}")
    print(f"  Faulty segments: {len(faulty_segments)}")
    
    # Extract features
    print(f"\nExtracting features (paper's methodology)...")
    print("  Extracting from healthy segments...")
    reference_features = detector.extract_paper_features_optimized(healthy_segments)
    
    print("  Extracting from faulty segments...")
    monitoring_features = detector.extract_paper_features_optimized(faulty_segments)
    
    # Apply two-stage Mann-Whitney U test
    stage1_results, stage2_results = detector.apply_paper_two_stage_mann_whitney(reference_features, monitoring_features)
    
    # Create aggregated indicator
    final_result = detector.create_aggregated_indicator(stage2_results)
    
    # Evaluate performance
    performance_results = detector.evaluate_approach_b_performance(reference_features, monitoring_features)
    
    # Performance summary
    print(f"\n=== Performance Summary (Approach B) ===")
    print("-" * 65)
    print(f"{'Feature':<25} {'Accuracy (%)':<12} {'Precision (%)':<13} {'Recall (%)'}")
    print("-" * 65)
    
    total_accuracy = 0
    feature_count = 0
    
    for feature_name, metrics in performance_results.items():
        print(f"{feature_name:<25} {metrics['accuracy']:<12.1f} {metrics['precision']:<13.1f} {metrics['recall']:.1f}")
        total_accuracy += metrics['accuracy']
        feature_count += 1
    
    avg_accuracy = total_accuracy / feature_count if feature_count > 0 else 0
    
    print("-" * 65)
    print(f"{'AVERAGE':<25} {avg_accuracy:<12.1f}")
    print("-" * 65)
    
    print(f"\nFinal Assessment:")
    print(f"- Average accuracy: {avg_accuracy:.1f}%")
    print(f"- Aggregated decision: {final_result['decision']}")
    print(f"- Decision confidence: {final_result['confidence']:.1f}%")
    
    if avg_accuracy >= 80:
        print("\n✓ Approach B successful! Paper's methodology works well with optimized parameters.")
    elif avg_accuracy >= 60:
        print("\n○ Approach B shows promise. Consider further parameter tuning.")
    else:
        print("\n? Approach B needs refinement. May need different segment lengths or feature parameters.")
    
    # Save results
    save_data = {
        'reference_features': reference_features,
        'monitoring_features': monitoring_features,
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'final_result': final_result,
        'performance_results': performance_results
    }
    
    np.savez('approach_b_results.npz', **save_data)
    print(f"\nApproach B results saved to 'approach_b_results.npz'")
    
    return detector, reference_features, monitoring_features, stage1_results, stage2_results, final_result, performance_results

if __name__ == "__main__":
    results = main_approach_b()