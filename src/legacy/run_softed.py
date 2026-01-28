
# Runs the code found here:
# https://github.com/cefet-rj-dal/reproduce/tree/main/softed


import numpy as np
import time
# from rpy2.robjects import r, numpy2ri

# Load required R scripts from softed_metrics directory
r('source("/home/sylli/Documents/ff-vus/src/softed_metrics/metrics.r")')
r('source("/home/sylli/Documents/ff-vus/src/softed_metrics/hard_metrics.r")')
r('source("/home/sylli/Documents/ff-vus/src/softed_metrics/softed_metrics.r")')

def debug_soft_scores(detection, event, k=15):
    """
    Debug function to see the raw soft scores at different k values
    """
    r_detection = numpy2ri.py2rpy(np.array(detection, dtype=bool))
    r_event = numpy2ri.py2rpy(np.array(event, dtype=bool))
    
    r_scores = r.soft_scores(r_detection, r_event, k=k)
    soft_scores_array = np.array(r_scores)
    
    return soft_scores_array


def visualize_soft_metric_mechanism(detection, event, k=15):
    """
    Visualize how the soft metric tolerance window works
    
    Shows:
    1. Detection and event signals
    2. Tolerance windows around each true event
    3. Which detections fall within each window
    4. The soft scores calculated
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Get indices
    event_indices = np.where(event)[0]
    detection_indices = np.where(detection)[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    
    # Plot 1: Signals with tolerance windows
    ax = axes[0]
    ax.set_title(f'Time Series with Tolerance Windows (k={k})', fontsize=12, fontweight='bold')
    
    # Plot the time series
    x = np.arange(len(event))
    ax.plot(x, detection.astype(int) * 0.5, 'b-', linewidth=1, alpha=0.3, label='Detection signal')
    ax.plot(x, event.astype(int) * 0.5, 'r-', linewidth=1, alpha=0.3, label='Event signal')
    
    # Plot event points
    ax.scatter(event_indices, np.ones(len(event_indices)) * 0.5, color='red', s=100, marker='v', 
               label='True Events', zorder=5)
    
    # Plot detection points
    ax.scatter(detection_indices, np.ones(len(detection_indices)) * 0.25, color='blue', s=100, marker='^',
               label='Detections', zorder=5)
    
    # Draw tolerance windows around each event
    for i, event_idx in enumerate(event_indices):
        # Window extends from [event-k, event+k]
        window_start = max(0, event_idx - k)
        window_end = min(len(event) - 1, event_idx + k)
        
        # Draw rectangle
        rect = patches.Rectangle((window_start, -0.05), window_end - window_start, 0.6,
                                linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.1,
                                linestyle='--', label='Tolerance window' if i == 0 else '')
        ax.add_patch(rect)
    
    ax.set_ylim(-0.1, 0.7)
    ax.set_xlabel('Sample Index')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Detailed view - Which detections match which events
    ax = axes[1]
    ax.set_title('Event-Detection Matching within Windows', fontsize=12, fontweight='bold')
    
    # For each event, show which detections are within its window
    y_pos = 0
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(event_indices), len(detection_indices))))
    
    for i, event_idx in enumerate(event_indices):
        window_start = max(0, event_idx - k)
        window_end = min(len(event) - 1, event_idx + k)
        
        # Find detections within this window
        detections_in_window = [d for d in detection_indices if window_start <= d <= window_end]
        
        # Draw the event
        ax.scatter(event_idx, y_pos, color='red', s=200, marker='v', zorder=5)
        ax.text(event_idx, y_pos - 0.3, f'Event {i}', ha='center', fontsize=9)
        
        # Draw the window
        ax.plot([window_start, window_end], [y_pos, y_pos], 'r--', linewidth=2, alpha=0.5)
        
        # Draw detections in this window
        for d in detections_in_window:
            ax.scatter(d, y_pos, color='blue', s=100, marker='o', zorder=4, alpha=0.7)
            # Draw a line connecting them
            ax.plot([event_idx, d], [y_pos, y_pos], 'g-', linewidth=1, alpha=0.3)
        
        y_pos += 1
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Event Index')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-5, len(event) + 5)
    
    # Plot 3: Soft scores returned
    ax = axes[2]
    soft_scores = debug_soft_scores(detection, event, k=k)
    
    # Only plot for detections that have non-zero scores
    non_zero_detections = [(detection_indices[i], soft_scores[i]) for i in range(len(soft_scores)) 
                          if soft_scores[i] > 0]
    
    if non_zero_detections:
        det_idx, det_scores = zip(*non_zero_detections)
        colors_scores = ['green' if s == 1.0 else 'orange' for s in det_scores]
        ax.bar(range(len(det_idx)), det_scores, color=colors_scores, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(det_idx)))
        ax.set_xticklabels([f'Det@{int(d)}' for d in det_idx], rotation=45)
    
    ax.set_ylabel('Soft Score (0-1)')
    ax.set_title('Soft Scores: How well each detection matches its closest event', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    ax.text(0.02, 0.98, 'Green = Perfect match (score=1.0)\nOrange = Partial match (0 < score < 1)',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def run_soft_metrics(detection, event, k=15, beta=1):
    """
    Calculate soft metrics using R implementation
    
    Parameters:
    -----------
    detection : numpy.ndarray or list
        Boolean array of detected events
    event : numpy.ndarray or list
        Boolean array of ground truth events
    k : int
        Tolerance window size (default: 15)
    beta : float
        Beta parameter for F-score (default: 1)
    
    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    float : Execution time in seconds
    """
    # Convert to R logical vectors
    r_detection = numpy2ri.py2rpy(np.array(detection, dtype=bool))
    r_event = numpy2ri.py2rpy(np.array(event, dtype=bool))

    # Time R implementation
    start_r = time.time()
    r_result = r.soft_metrics(r_detection, r_event, k=k, beta=beta)
    time_r = time.time() - start_r

    # Extract R results
    r_metrics = {
        'TP': float(r_result.rx2('TP')[0]),
        'FP': float(r_result.rx2('FP')[0]),
        'FN': float(r_result.rx2('FN')[0]),
        'TN': float(r_result.rx2('TN')[0]),
        'accuracy': float(r_result.rx2('accuracy')[0]),
        'sensitivity': float(r_result.rx2('sensitivity')[0]),
        'specificity': float(r_result.rx2('specificity')[0]),
        'precision': float(r_result.rx2('precision')[0]),
        'recall': float(r_result.rx2('recall')[0]),
        'F1': float(r_result.rx2('F1')[0]),
        'PPV': float(r_result.rx2('PPV')[0]),
        'NPV': float(r_result.rx2('NPV')[0]),
        'prevalence': float(r_result.rx2('prevalence')[0]),
        'detection_rate': float(r_result.rx2('detection_rate')[0]),
        'detection_prevalence': float(r_result.rx2('detection_prevalence')[0]),
        'balanced_accuracy': float(r_result.rx2('balanced_accuracy')[0]),
    }
    
    return r_metrics, time_r


def run_hard_metrics(detection, event, beta=1):
    """
    Calculate hard metrics using R implementation
    
    Parameters:
    -----------
    detection : numpy.ndarray or list
        Boolean array of detected events
    event : numpy.ndarray or list
        Boolean array of ground truth events
    beta : float
        Beta parameter for F-score (default: 1)
    
    Returns:
    --------
    dict : Dictionary containing all calculated metrics
    float : Execution time in seconds
    """
    # Convert to R logical vectors
    r_detection = numpy2ri.py2rpy(np.array(detection, dtype=bool))
    r_event = numpy2ri.py2rpy(np.array(event, dtype=bool))
    
    # Time R implementation
    start_r = time.perf_counter()
    r_result = r.hard_metrics(r_detection, r_event, beta=beta)
    time_r = time.perf_counter() - start_r
    
    # Extract R results
    r_metrics = {
        'TP': float(r_result.rx2('TP')[0]),
        'FP': float(r_result.rx2('FP')[0]),
        'FN': float(r_result.rx2('FN')[0]),
        'TN': float(r_result.rx2('TN')[0]),
        'accuracy': float(r_result.rx2('accuracy')[0]),
        'sensitivity': float(r_result.rx2('sensitivity')[0]),
        'specificity': float(r_result.rx2('specificity')[0]),
        'precision': float(r_result.rx2('precision')[0]),
        'recall': float(r_result.rx2('recall')[0]),
        'F1': float(r_result.rx2('F1')[0]),
        'PPV': float(r_result.rx2('PPV')[0]),
        'NPV': float(r_result.rx2('NPV')[0]),
        'prevalence': float(r_result.rx2('prevalence')[0]),
        'detection_rate': float(r_result.rx2('detection_rate')[0]),
        'detection_prevalence': float(r_result.rx2('detection_prevalence')[0]),
        'balanced_accuracy': float(r_result.rx2('balanced_accuracy')[0]),
    }
    
    return r_metrics, time_r



# Example usage
if __name__ == "__main__":
    print("="*70)
    print("SOFTED METRIC VISUALIZATION AND ANALYSIS")
    print("="*70)
    
    # Create simple example with clear patterns
    print("\nGenerating sample data with clear detection/event patterns...\n")
    
    # Create a simple scenario: 100-sample time series
    detection = np.zeros(100, dtype=bool)
    event = np.zeros(100, dtype=bool)
    
    # True events at specific locations
    true_event_locs = [20, 50, 80]
    for loc in true_event_locs:
        event[loc] = True
    
    # Detections slightly offset from true events
    detection_locs = [18, 22, 48, 52, 78, 85]  # Some perfect, some off, some false positives
    for loc in detection_locs:
        detection[loc] = True
    
    print(f"True Events at indices: {true_event_locs}")
    print(f"Detections at indices: {detection_locs}")
    print(f"\nNote: Some detections are offset from true events")
    
    # Visualize with different k values
    print("\n" + "="*70)
    print("VISUALIZING HOW TOLERANCE WINDOW (k) AFFECTS MATCHING")
    print("="*70)
    
    import matplotlib.pyplot as plt
    
    k_values_to_show = [5, 10, 20]
    
    for k in k_values_to_show:
        print(f"\n--- Window size k={k} ---")
        
        # Get metrics
        soft_result, _ = run_soft_metrics(detection, event, k=k)
        
        print(f"  TP (True Positives):  {soft_result['TP']:.2f}")
        print(f"  FP (False Positives): {soft_result['FP']:.2f}")
        print(f"  FN (False Negatives): {soft_result['FN']:.2f}")
        print(f"  Precision: {soft_result['precision']:.4f}")
        print(f"  Recall:    {soft_result['recall']:.4f}")
        print(f"  F1:        {soft_result['F1']:.4f}")
        
        # Create visualization
        fig = visualize_soft_metric_mechanism(detection, event, k=k)
        plt.savefig(f'/tmp/soft_metric_k{k}.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  → Visualization saved to /tmp/soft_metric_k{k}.png")
    
    print("\n" + "="*70)
    print("EXPLANATION OF HOW SOFTED METRIC WORKS:")
    print("="*70)
    print("""
1. TOLERANCE WINDOW:
   - For each true event at position E, create a window: [E-k, E+k]
   - This is a "soft" region around the true event
   
2. MATCHING DETECTION TO EVENTS:
   - For each detection at position D, find the closest event E
   - Check if D falls within that event's window [E-k, E+k]
   
3. SOFT SCORE CALCULATION:
   - If D is far from window → score = 0 (False Positive)
   - If D is at edge of window → score = 0.5 (Partial match)
   - If D is at center of window → score = 1.0 (Perfect match)
   
4. TP/FP COUNTING:
   - Instead of counting discrete TP/FP, you sum the soft scores
   - This gives fractional true positives (TPs = 35.3 instead of just 35)
   
5. WHY IT'S "MORE TOLERANT":
   - Hard metrics: Detection off by 1 sample = WRONG
   - Soft metrics: Detection off by k samples = PARTIALLY RIGHT
   - Bigger k = more tolerance = higher recall, different F1 score
    """)
    
    print("\nWith different k values:")
    print("  - k=5:  Very strict window → lower recall")
    print("  - k=15: Moderate window → balanced")
    print("  - k=30: Very forgiving window → higher recall")

