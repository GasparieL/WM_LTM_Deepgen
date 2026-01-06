"""
Plotting P-Values and P-Values Flipped

Clean plotting functions for visualizing model fits and inverted model fits
from both WM and LTM extraction methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_wm_p_values(csv_file='df_total_p_pilot_total_TEST.csv', save_fig=False):
    """
    Plot WM p-values and p-values flipped.

    Args:
        csv_file: Path to WM data CSV
        save_fig: Whether to save figure to file
    """
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # IT - P values (model fit)
    axes[0, 0].scatter(df['x_values_it'], df['p_values_it'],
                       s=1, alpha=0.5, c='#49a791')
    axes[0, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[0, 0].set_xlabel('IT Similarity')
    axes[0, 0].set_ylabel('P (Model Fit)')
    axes[0, 0].set_title('WM - IT Model Fit')
    axes[0, 0].grid(alpha=0.3)

    # IT - P flipped (inverted model fit)
    axes[0, 1].scatter(df['x_values_it'], df['p_values_it_flipped'],
                       s=1, alpha=0.5, c='#e76f51')
    axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[0, 1].set_xlabel('IT Similarity')
    axes[0, 1].set_ylabel('P Flipped (Inverted Model Fit)')
    axes[0, 1].set_title('WM - IT Inverted Model Fit')
    axes[0, 1].grid(alpha=0.3)

    # V2 - P values (model fit)
    axes[1, 0].scatter(df['x_values_v2'], df['p_values_v2'],
                       s=1, alpha=0.5, c='#95c355')
    axes[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[1, 0].set_xlabel('V2 Similarity')
    axes[1, 0].set_ylabel('P (Model Fit)')
    axes[1, 0].set_title('WM - V2 Model Fit')
    axes[1, 0].grid(alpha=0.3)

    # V2 - P flipped (inverted model fit)
    axes[1, 1].scatter(df['x_values_v2'], df['p_values_v2_flipped'],
                       s=1, alpha=0.5, c='#e76f51')
    axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[1, 1].set_xlabel('V2 Similarity')
    axes[1, 1].set_ylabel('P Flipped (Inverted Model Fit)')
    axes[1, 1].set_title('WM - V2 Inverted Model Fit')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig('wm_p_values.png', dpi=300, bbox_inches='tight')
        print("Saved: wm_p_values.png")

    plt.show()


def plot_ltm_p_values(csv_file='enhanced_df.csv', save_fig=False):
    """
    Plot LTM p-values and p-values flipped.

    Args:
        csv_file: Path to LTM data CSV
        save_fig: Whether to save figure to file
    """
    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # IT - P values (model fit)
    axes[0, 0].scatter(df['LTM - NonLTM IT Distractor Similarity'],
                       df['p_values_it_ltm'],
                       s=1, alpha=0.5, c='#49a791')
    axes[0, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[0, 0].set_xlabel('IT Similarity')
    axes[0, 0].set_ylabel('P (Model Fit)')
    axes[0, 0].set_title('LTM - IT Model Fit')
    axes[0, 0].grid(alpha=0.3)

    # IT - P flipped (inverted model fit)
    axes[0, 1].scatter(df['LTM - NonLTM IT Distractor Similarity'],
                       df['p_values_it_flipped_ltm'],
                       s=1, alpha=0.5, c='#e76f51')
    axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[0, 1].set_xlabel('IT Similarity')
    axes[0, 1].set_ylabel('P Flipped (Inverted Model Fit)')
    axes[0, 1].set_title('LTM - IT Inverted Model Fit')
    axes[0, 1].grid(alpha=0.3)

    # V2 - P values (model fit)
    axes[1, 0].scatter(df['LTM - NonLTM V2 Distractor Similarity'],
                       df['p_values_v2_ltm'],
                       s=1, alpha=0.5, c='#95c355')
    axes[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[1, 0].set_xlabel('V2 Similarity')
    axes[1, 0].set_ylabel('P (Model Fit)')
    axes[1, 0].set_title('LTM - V2 Model Fit')
    axes[1, 0].grid(alpha=0.3)

    # V2 - P flipped (inverted model fit)
    axes[1, 1].scatter(df['LTM - NonLTM V2 Distractor Similarity'],
                       df['p_values_v2_flipped_ltm'],
                       s=1, alpha=0.5, c='#e76f51')
    axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[1, 1].set_xlabel('V2 Similarity')
    axes[1, 1].set_ylabel('P Flipped (Inverted Model Fit)')
    axes[1, 1].set_title('LTM - V2 Inverted Model Fit')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig('ltm_p_values.png', dpi=300, bbox_inches='tight')
        print("Saved: ltm_p_values.png")

    plt.show()


def plot_combined_comparison(wm_file='df_total_p_pilot_total_TEST.csv',
                            ltm_file='enhanced_df.csv',
                            save_fig=False):
    """
    Plot p and p_flipped overlaid to show mirror symmetry.

    Args:
        wm_file: Path to WM data CSV
        ltm_file: Path to LTM data CSV
        save_fig: Whether to save figure to file
    """
    df_wm = pd.read_csv(wm_file)
    df_ltm = pd.read_csv(ltm_file)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # WM IT
    axes[0, 0].scatter(df_wm['x_values_it'], df_wm['p_values_it'],
                       s=1, alpha=0.5, c='#49a791', label='P (model fit)')
    axes[0, 0].scatter(df_wm['x_values_it'], df_wm['p_values_it_flipped'],
                       s=1, alpha=0.5, c='#e76f51', label='P flipped (inverted)')
    axes[0, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.3, label='y=0.5')
    axes[0, 0].set_xlabel('IT Similarity')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('WM - IT: P vs P Flipped')
    axes[0, 0].legend(markerscale=5)
    axes[0, 0].grid(alpha=0.3)

    # WM V2
    axes[0, 1].scatter(df_wm['x_values_v2'], df_wm['p_values_v2'],
                       s=1, alpha=0.5, c='#95c355', label='P (model fit)')
    axes[0, 1].scatter(df_wm['x_values_v2'], df_wm['p_values_v2_flipped'],
                       s=1, alpha=0.5, c='#e76f51', label='P flipped (inverted)')
    axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.3, label='y=0.5')
    axes[0, 1].set_xlabel('V2 Similarity')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('WM - V2: P vs P Flipped')
    axes[0, 1].legend(markerscale=5)
    axes[0, 1].grid(alpha=0.3)

    # LTM IT
    axes[1, 0].scatter(df_ltm['LTM - NonLTM IT Distractor Similarity'],
                       df_ltm['p_values_it_ltm'],
                       s=1, alpha=0.5, c='#49a791', label='P (model fit)')
    axes[1, 0].scatter(df_ltm['LTM - NonLTM IT Distractor Similarity'],
                       df_ltm['p_values_it_flipped_ltm'],
                       s=1, alpha=0.5, c='#e76f51', label='P flipped (inverted)')
    axes[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(0.5, color='gray', linestyle=':', alpha=0.3, label='y=0.5')
    axes[1, 0].set_xlabel('IT Similarity')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('LTM - IT: P vs P Flipped')
    axes[1, 0].legend(markerscale=5)
    axes[1, 0].grid(alpha=0.3)

    # LTM V2
    axes[1, 1].scatter(df_ltm['LTM - NonLTM V2 Distractor Similarity'],
                       df_ltm['p_values_v2_ltm'],
                       s=1, alpha=0.5, c='#95c355', label='P (model fit)')
    axes[1, 1].scatter(df_ltm['LTM - NonLTM V2 Distractor Similarity'],
                       df_ltm['p_values_v2_flipped_ltm'],
                       s=1, alpha=0.5, c='#e76f51', label='P flipped (inverted)')
    axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.3, label='y=0.5')
    axes[1, 1].set_xlabel('V2 Similarity')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('LTM - V2: P vs P Flipped')
    axes[1, 1].legend(markerscale=5)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig('p_vs_p_flipped_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: p_vs_p_flipped_comparison.png")

    plt.show()


def verify_sum_equals_one(wm_file='df_total_p_pilot_total_TEST.csv',
                          ltm_file='enhanced_df.csv'):
    """
    Verify that p + p_flipped = 1.0 for all data.

    Args:
        wm_file: Path to WM data CSV
        ltm_file: Path to LTM data CSV
    """
    print("Verifying: p + p_flipped = 1.0")
    print("=" * 50)

    # WM verification
    df_wm = pd.read_csv(wm_file)

    wm_it_sum = df_wm['p_values_it'] + df_wm['p_values_it_flipped']
    wm_v2_sum = df_wm['p_values_v2'] + df_wm['p_values_v2_flipped']

    print("\nWM Data:")
    print(f"  IT: mean={wm_it_sum.mean():.10f}, std={wm_it_sum.std():.10f}")
    print(f"  V2: mean={wm_v2_sum.mean():.10f}, std={wm_v2_sum.std():.10f}")

    # LTM verification
    df_ltm = pd.read_csv(ltm_file)

    ltm_it_sum = df_ltm['p_values_it_ltm'] + df_ltm['p_values_it_flipped_ltm']
    ltm_v2_sum = df_ltm['p_values_v2_ltm'] + df_ltm['p_values_v2_flipped_ltm']

    print("\nLTM Data:")
    print(f"  IT: mean={ltm_it_sum.mean():.10f}, std={ltm_it_sum.std():.10f}")
    print(f"  V2: mean={ltm_v2_sum.mean():.10f}, std={ltm_v2_sum.std():.10f}")

    # Check if all are 1.0
    all_correct = all([
        abs(wm_it_sum.mean() - 1.0) < 1e-6,
        abs(wm_v2_sum.mean() - 1.0) < 1e-6,
        abs(ltm_it_sum.mean() - 1.0) < 1e-6,
        abs(ltm_v2_sum.mean() - 1.0) < 1e-6
    ])

    print("\n" + "=" * 50)
    if all_correct:
        print("Result: All sums equal 1.0 (verified)")
    else:
        print("Result: WARNING - Some sums do not equal 1.0")
    print("=" * 50)


if __name__ == "__main__":
    """Run all plotting functions"""

    print("Plotting WM p-values...")
    plot_wm_p_values(save_fig=True)

    print("\nPlotting LTM p-values...")
    plot_ltm_p_values(save_fig=True)

    print("\nPlotting combined comparison...")
    plot_combined_comparison(save_fig=True)

    print("\nVerifying p + p_flipped = 1.0...")
    verify_sum_equals_one()

    print("\nAll plots complete!")
