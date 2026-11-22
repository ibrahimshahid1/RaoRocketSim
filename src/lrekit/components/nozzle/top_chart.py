"""Rao/TOP angles measured from NASA SP-125 Figure 4-14 (PDF p. 100).

These are reviewer digitizations, not source-printed numerical tables.  The
single printed worked point on PDF p. 107 is retained exactly.  Historical raw
measurements and their scripts live in the provenance artifacts named below.
"""
from __future__ import annotations

import numpy as np

EPSILON = np.array([5, 6, 8, 10, 14, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)
LENGTH_PERCENT = np.array([60, 70, 80, 90, 100], dtype=float)
THETA_N_DEG = np.array([[28.83, 25.69, 23.46, 21.62, 20.44], [29.29, 26.0, 23.76, 21.92, 20.66], [30.8, 27.26, 24.82, 23.01, 21.66], [31.91, 28.23, 25.74, 23.93, 22.49], [33.59, 29.87, 27.4, 25.5, 23.98], [33.99, 30.23, 27.71, 25.84, 24.34], [35.14, 31.44, 28.91, 27.06, 25.51], [36.39, 32.47, 29.91, 28.03, 26.6], [37.17, 33.19, 30.6, 28.71, 27.34], [37.8, 33.88, 31.24, 29.37, 28.09], [38.29, 34.43, 31.74, 29.97, 28.59], [38.73, 34.94, 32.18, 30.34, 29.11], [39.02, 35.29, 32.53, 30.86, 29.53]], dtype=float)
THETA_E_DEG = np.array([[19.62, 15.94, 12.91, 10.03, 8.15], [19.12, 15.45, 12.5, 9.88, 7.94], [17.75, 14.16, 11.34, 9.05, 7.27], [16.7, 13.21, 10.49, 8.37, 6.67], [15.45, 12.06, 9.8, 7.49, 5.86], [15.15, 11.88, 9.41, 7.34, 5.7], [14.52, 11.24, 8.92, 6.75, 5.16], [14.04, 10.8, 8.59, 6.64, 4.97], [13.63, 10.49, 8.36, 6.57, 4.97], [13.37, 10.23, 8.25, 6.57, 4.97], [13.11, 10.12, 8.15, 6.57, 4.97], [12.99, 10.12, 8.15, 6.57, 4.97], [12.73, 10.0, 8.12, 6.57, 4.97]], dtype=float)

EVIDENCE = {'chart_id': 'sp125-fig4-14:v2', 'model_id': 'MODEL_NOZZLE_RAO_TOP_BEZIER', 'document_id': 'DOC_NTRS_19710019929', 'source_sha256': '471e590a0c8e99c36a98132104a3ddcc7097b0b7c561fb2638ce9d3c80e26c87', 'figure_pdf_page': 100, 'worked_example_pdf_page': 107, 'measurement_kind': 'reviewer_digitized_original_chart', 'digitization_file': 'knowledge/workflows/results/agentic-2026-10-16/p1/sp125_fig4_14_digitized.json', 'digitization_sha256': '2e51c0522244477a03622d63c5607d80afc2bf256a7b15d23a21b735bad35a26', 'comparison_file': 'knowledge/workflows/results/agentic-2026-10-16/P1-rao-top-angle-chart-comparison.json', 'reading_uncertainty_deg': 0.3, 'reading_uncertainty_near_epsilon_5_6_deg': 0.6, 'interpolation_error': 'NOT_ESTABLISHED_between_sampled_knots', 'processing': 'Linear interpolation in epsilon and percent length; source-stated worked point14/80 overrides measured27.36/9.64 with27.4/9.8; cumulative-minimum regularization suppresses <=0.16deg exit-angle scan fluctuations above epsilon25. No chart extrapolation.', 'physical_validation': 'NOT_ESTABLISHED', 'secondary_comparison': 'SP-8120 Fig5b agrees in trend, differs up to~1.5deg initial angle; not averaged into this table.'}
