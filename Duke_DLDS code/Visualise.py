import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Keep import as is
import pandas as pd

# Set paths
root_dir = r"C:\Softwares\All Programs\HCA\Duke_DLDS\Series_Classification"
csv_path = r"C:\Softwares\All Programs\HCA\Duke_DLDS\SeriesClassificationKey.csv"

def get_sample_images():
    # Read the classification key
    df = pd.read_csv(csv_path)
    unique_labels = sorted(df['Label'].unique())

    # Create a figure with better dimensions
    num_labels = len(unique_labels)
    cols = 4
    rows = (num_labels + cols - 1) // cols

    # Set up the plot style
    plt.style.use('dark_background')
    # Try increasing vertical space slightly more if needed
    plt.rcParams['figure.figsize'] = [20, 6.5*rows] # Slightly more height per row
    plt.rcParams['figure.dpi'] = 100

    # Create figure and axes - IMPORTANT: capture the 'fig' object
    fig, axes = plt.subplots(rows, cols)
    axes = axes.ravel() # Flatten axes array

    # --- Loop to plot images ---
    last_successful_idx = -1 # Keep track of last plotted index
    for idx, label in enumerate(unique_labels):
        if idx >= len(axes):
            break

        # Get first occurrence of this label
        try:
            row = df[df['Label'] == label].iloc[0]
            patient_id = f"{int(row['DLDS']):04d}"
            series_id = str(row['Series'])
            series_path = os.path.join(root_dir, patient_id, series_id)

            if os.path.exists(series_path):
                files = [f for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))]
                if files:
                    try:
                        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
                    except ValueError:
                        files.sort()

                    middle_idx = len(files) // 2
                    img_path = os.path.join(series_path, files[middle_idx])
                    ds = pydicom.dcmread(img_path, force=True)

                    if hasattr(ds, 'PixelData'):
                        img = ds.pixel_array
                        p2, p98 = np.percentile(img, (2, 98))
                        img = np.clip(img, p2, p98)
                        if (p98 - p2) > 0:
                            img = (img - p2) / (p98 - p2)
                        else:
                            img = np.zeros_like(img)

                        axes[idx].imshow(img, cmap='gray')
                        axes[idx].axis('off')

                        # --- Text Adjustment ---
                        # Move text slightly closer to the axis
                        axes[idx].text(0.5, -0.08, f'Label {label}', # Changed y from -0.1 to -0.08
                                       horizontalalignment='center',
                                       verticalalignment='top',
                                       transform=axes[idx].transAxes,
                                       color='white',
                                       fontsize=12,
                                       bbox=dict(facecolor='black',
                                                 edgecolor='none',
                                                 alpha=0.8,
                                                 pad=3))
                        last_successful_idx = idx # Update last successful index
                    else:
                        axes[idx].text(0.5, 0.5, 'No PixelData', color='red', ha='center', va='center')
                        axes[idx].axis('off')
                else:
                    axes[idx].text(0.5, 0.5, 'No Files', color='red', ha='center', va='center')
                    axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, 'Path Not Found', color='red', ha='center', va='center')
                axes[idx].axis('off')
        except IndexError:
            axes[idx].text(0.5, 0.5, f'Label {label}\n(No Data)', color='orange', ha='center', va='center')
            axes[idx].axis('off')
        except Exception as e:
            print(f"Error processing label {label} in series {series_path if 'series_path' in locals() else 'N/A'}: {e}")
            axes[idx].text(0.5, 0.5, f'Error\nLabel {label}', color='red', ha='center', va='center')
            axes[idx].axis('off')
    # --- End of Loop ---


    # Turn off any unused subplots
    for i in range(last_successful_idx + 1, len(axes)):
         axes[i].axis('off')

    # Add main title
    plt.suptitle('Sample Images for Each Label Category',
                 fontsize=16,
                 color='white')

    # --- Layout Adjustments ---
    # Method 1: Use fig.subplots_adjust (often more reliable for explicit margins)
    # Increase bottom margin significantly (e.g., to 0.12 or 12%)
    # Also adjust top slightly to ensure suptitle fits
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, hspace=0.5, wspace=0.3)

    # Method 2: Use tight_layout with rect (Alternative to Method 1)
    # If Method 1 doesn't work well, comment it out and uncomment this:
    # plt.tight_layout(rect=[0, 0.1, 1, 0.93]) # Increased bottom margin in rect

    # Note: Avoid using both fig.subplots_adjust for bottom/top/left/right AND tight_layout(rect=...)
    # as they might conflict. You can use fig.subplots_adjust for hspace/wspace
    # and then plt.tight_layout() *without* rect if needed for general overlap prevention.


    # Save figure
    plt.savefig('label_samples.png',
                dpi=300,
                # Critical Point: If still cut off, 'bbox_inches=tight' might be the culprit.
                # Try removing it or setting bbox_inches=None.
                bbox_inches='tight',
                facecolor='black',
                edgecolor='none')

    # Display figure
    plt.show()

if __name__ == "__main__":
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found at {root_dir}")
    elif not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
    else:
        get_sample_images()