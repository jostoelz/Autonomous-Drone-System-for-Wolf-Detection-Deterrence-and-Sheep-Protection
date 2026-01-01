import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- SETTINGS ---
# Use a context suited for papers (larger fonts, distinct lines)
# This ensures the charts look good when pasted into a Word document or LaTeX.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300 # High resolution for print quality

def generate_plots(csv_file):
    # Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Make sure to run the benchmark first.")
        return

    # Define consistent colors for models across ALL charts
    # Using specific hex codes ensures YOLO is always Blue and GPT is always Red.
    model_colors = {"YOLOv8": "#3498db", "GPT-4o": "#e74c3c"} 

    # Prepare Data for IoU Plots
    # We 'melt' the dataframe to make it compatible with Seaborn's long-form data expectation.
    plot_data = df.melt(value_vars=['yolo_iou', 'gpt_iou'], 
                        var_name='Model', value_name='IoU')
    
    # Rename the columns to readable names for the legend
    plot_data['Model'] = plot_data['Model'].replace({'yolo_iou': 'YOLOv8', 'gpt_iou': 'GPT-4o'})

    # ---------------------------------------------------------
    # 1a. IoU DISTRIBUTION (Combined Boxplot + Stripplot)
    # ---------------------------------------------------------
    # This plot is scientifically preferred because it shows both the statistical summary 
    # (medians/quartiles) AND the raw data density.
    plt.figure(figsize=(8, 6))
    
    # Draw the Boxplot to show the quartiles. 
    # We hide outliers ('showfliers=False') because we will overlay the actual points anyway.
    sns.boxplot(data=plot_data, x='Model', y='IoU', palette=model_colors, 
                showfliers=False, boxprops={'alpha': 0.6})
    
    # Overlay a Stripplot to show the actual data points.
    # This reveals if the data is clustered or spread out.
    sns.stripplot(data=plot_data, x='Model', y='IoU', color='black', 
                  alpha=0.3, jitter=True, size=3)

    plt.title('Spatial Precision: Boxplot & Raw Data', fontweight='bold')
    plt.ylabel('Intersection over Union (IoU)')
    plt.xlabel('') # The labels on the x-axis are self-explanatory
    plt.ylim(0, 1.05)
    plt.tight_layout() # Adjust layout to prevent clipping of labels
    plt.savefig('1a_iou_box_strip.png')
    plt.close() # Close the figure to free up memory
    print("Generated: 1a IoU Box/Strip Plot")

    # ---------------------------------------------------------
    # 1b. IoU DENSITY (Violinplot)
    # ---------------------------------------------------------
    # Visualizes the probability density of the data. 
    # Good for seeing the "shape" of the results distribution.
    plt.figure(figsize=(8, 6))

    # 'cut=0' ensures the violin doesn't visually extend past the actual min/max data values.
    # 'inner="quartile"' draws dashed lines inside the violin to indicate the 25%, 50%, and 75% marks.
    sns.violinplot(data=plot_data, x='Model', y='IoU', palette=model_colors, 
                   inner="quartile", cut=0, linewidth=1.5)

    plt.title('Spatial Precision: Density Distribution (Violinplot)', fontweight='bold')
    plt.ylabel('Intersection over Union (IoU)')
    plt.xlabel('')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('1b_iou_violinplot.png')
    plt.close()
    print("Generated: 1b IoU Violinplot")

    # ---------------------------------------------------------
    # 2. RECALL BY CLASS (With Confidence Intervals)
    # ---------------------------------------------------------
    # We calculate how often each object class was successfully found.
    # Reshape the data to list every detection attempt as a row.
    recall_data = df.melt(id_vars=['class'], value_vars=['yolo_found', 'gpt_found'], 
                          var_name='Model', value_name='Found')
    recall_data['Model'] = recall_data['Model'].replace({'yolo_found': 'YOLOv8', 'gpt_found': 'GPT-4o'})
    recall_data['Found'] = recall_data['Found'] * 100 # Convert 0/1 to Percentage (0% or 100%)

    plt.figure(figsize=(12, 6))
    
    # The 'errorbar' parameter automatically calculates the 95% Confidence Interval (CI).
    # This shows the statistical reliability of the mean recall.
    sns.barplot(data=recall_data, x='class', y='Found', hue='Model', 
                palette=model_colors, errorbar=('ci', 95), capsize=0.1)
    
    plt.title('Detection Recall per Class (with 95% CI)', fontweight='bold')
    plt.ylabel('Recall / Detection Rate (%)')
    plt.xlabel('Object Class')
    plt.ylim(0, 105)
    plt.legend(title='Model', loc='lower right')
    plt.xticks(rotation=45) # Rotate labels if there are many classes
    plt.tight_layout()
    plt.savefig('2_recall_per_class.png')
    plt.close()
    print("Generated: 2. Recall by Class Plot")

    # ---------------------------------------------------------
    # 3. LATENCY COMPARISON (Log Scale + Stats)
    # ---------------------------------------------------------
    # Comparison of inference speed. Since GPT is much slower, a Log Scale is necessary.
    plt.figure(figsize=(8, 6))
    
    latency_data = df.melt(value_vars=['yolo_latency_ms', 'gpt_latency_ms'],
                           var_name='Model', value_name='Time_ms')
    latency_data['Model'] = latency_data['Model'].replace({'yolo_latency_ms': 'YOLOv8', 'gpt_latency_ms': 'GPT-4o'})

    # Create the bar plot with Standard Deviation error bars
    ax = sns.barplot(data=latency_data, x='Model', y='Time_ms', palette=model_colors, errorbar='sd', capsize=0.1)
    
    # Set y-axis to logarithmic scale to handle the massive difference in magnitudes
    ax.set_yscale("log")
    plt.title('Inference Latency Comparison (Log Scale)', fontweight='bold')
    plt.ylabel('Time (ms) - Logarithmic')
    plt.xlabel('')

    # Calculate means manually to place text labels accurately
    means = latency_data.groupby('Model')['Time_ms'].mean()
    for i, model in enumerate(['YOLOv8', 'GPT-4o']):
        mean_val = means[model]
        # We position the text slightly above the bar. 
        # On a log scale, multiplication works better than addition for offsets.
        ax.text(i, mean_val * 1.3, f"{mean_val:.1f} ms", 
                color='black', ha="center", fontweight='bold')

    plt.tight_layout()
    plt.savefig('3_latency_log.png')
    plt.close()
    print("Generated: 3. Latency Plot")

    # ---------------------------------------------------------
    # 4. SUMMARY STATISTICS (CSV)
    # ---------------------------------------------------------
    # Create a summary table for numerical analysis in the paper.
    summary = {
        "Metric": ["Mean IoU", "Total Objects (GT)", "Detected Objects", "Recall (%)", "Avg Latency (ms)"],
        "YOLOv8": [
            df['yolo_iou'].mean(),
            len(df), # Total rows equals total Ground Truth objects
            df['yolo_found'].sum(),
            (df['yolo_found'].mean() * 100),
            df['yolo_latency_ms'].mean()
        ],
        "GPT-4o": [
            df['gpt_iou'].mean(),
            len(df),
            df['gpt_found'].sum(),
            (df['gpt_found'].mean() * 100),
            df['gpt_latency_ms'].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary).round(4)
    summary_df.to_csv("summary_statistics.csv", index=False)
    print("Generated: Summary Statistics Table")

if __name__ == "__main__":
    # Point this to your actual results file
    csv_filename = "detailed_comparison.csv"
    generate_plots(csv_filename)
    print("\nProcessing complete. All diagrams have been successfully created.")
