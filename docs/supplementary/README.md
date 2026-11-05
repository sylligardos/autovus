# Supplementary results

Results left out of the paper for space reasons, with the text that accompanied them.

## Runtime as a function of the maximum buffer length

The buffer is one of the central concepts of VUS, so we analyze how the runtime of our measures behaves as the buffer length grows, measured on a single detector. The figure below reports our results. VUS (in red) is the slowest of the three, with one order of magnitude difference at buffer length 0 that grows to three orders of magnitude at buffer length 512, compared to FastVUS-GPU (in green). VUS is also the most affected by the increase in buffer length, starting at about 60 milliseconds per time series for buffer length 0 (no buffer) and increasing to about 10 seconds per time series for the maximum evaluated buffer. Then follows FastVUS, whose runtime remains almost constant up to buffer length 16. Finally, FastVUS-GPU behaves like FastVUS but is an order of magnitude faster, of the same order as AUC. Overall, both our approaches scale well with buffer length and remain practical for lengths up to 512.

![Runtime versus maximum buffer length](runtime_vs_buffer_length.png)

Average runtime per time series for VUS (red), FastVUS (yellow), and FastVUS-GPU (green) with respect to the buffer length (x-axis in log scale). The black dashed line is the average runtime of AUC. Buffer lengths 0, 2, 4, ..., 512. PDF: [runtime_vs_buffer_length.pdf](runtime_vs_buffer_length.pdf).

## Optional step parameter of FastVUS

FastVUS adds an optional *step* parameter that evaluates every k-th buffer rather than all of them, trading a small, controlled error for speed. More usefully, because FastVUS evaluates the measure per buffer, restricting it to a single buffer length (step equal to L) approximates other event-aware measures such as Range-AUC (Paparrizos et al., VLDB 2022) or SoftED (Salles et al., 2024), so the step parameter also lets the same machinery estimate a family of buffer-based metrics, not only VUS. This is the only approximation in FastVUS and it is off by default (`step=1`).
