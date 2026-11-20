import io
import textwrap
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Ensure headless rendering for cloud servers without display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_pdf_report(inputs, soh, recommendation):
    """
    Generate a professional single-page PDF report for battery health assessment.
    inputs: dict with keys 'cycle', 'v_mean', 'v_min', 't_mean', 't_max'
    soh: float (State of Health percentage)
    recommendation: dict returned by recommend(soh)
    """
    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Colors
    header_bg = "#0F172A"       # Dark Slate
    card_bg = "#F8FAFC"         # Light Gray-Blue
    card_border = "#E2E8F0"
    
    condition = recommendation.get("condition", "Unknown")
    if condition == "Healthy":
        cond_color = "#16A34A"  # Green
    elif condition == "Aging":
        cond_color = "#D97706"  # Amber
    else:
        cond_color = "#DC2626"  # Red

    # 1. Header Banner
    ax.add_patch(patches.Rectangle((0, 0.88), 1, 0.12, facecolor=header_bg))
    ax.text(0.08, 0.94, "VoltSense AI", fontsize=22, fontweight="bold", color="white")
    ax.text(0.08, 0.90, "Battery Health Assessment Report", fontsize=13, color="#94A3B8")
    
    # Generation Timestamp
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    ax.text(0.92, 0.90, f"Generated: {now_str}", fontsize=9, color="#CBD5E1", ha="right")

    # 2. Executive Summary Cards (SOH & Condition)
    ax.add_patch(patches.FancyBboxPatch((0.08, 0.73), 0.40, 0.12, boxstyle="round,pad=0.015",
                                        facecolor=card_bg, edgecolor=card_border, linewidth=1.2))
    ax.text(0.12, 0.81, "State of Health (SOH)", fontsize=11, color="#64748B", fontweight="bold")
    ax.text(0.12, 0.75, f"{soh:.1f}%", fontsize=26, color="#0F172A", fontweight="bold")

    ax.add_patch(patches.FancyBboxPatch((0.52, 0.73), 0.40, 0.12, boxstyle="round,pad=0.015",
                                        facecolor=card_bg, edgecolor=card_border, linewidth=1.2))
    ax.text(0.56, 0.81, "Battery Condition", fontsize=11, color="#64748B", fontweight="bold")
    ax.text(0.56, 0.75, condition, fontsize=24, color=cond_color, fontweight="bold")

    # 3. Input Parameters
    ax.text(0.08, 0.68, "1. Discharge Cycle & Sensor Inputs", fontsize=12, fontweight="bold", color="#1E293B")
    ax.plot([0.08, 0.92], [0.665, 0.665], color="#CBD5E1", lw=1)

    input_data = [
        ("Discharge Cycle Number", f"{inputs.get('cycle', 'N/A')}"),
        ("Mean Voltage", f"{inputs.get('v_mean', 0):.2f} V"),
        ("Minimum Voltage", f"{inputs.get('v_min', 0):.2f} V"),
        ("Mean Temperature", f"{inputs.get('t_mean', 0):.1f} °C"),
        ("Maximum Temperature", f"{inputs.get('t_max', 0):.1f} °C"),
    ]

    y_pos = 0.63
    for label, val in input_data:
        ax.text(0.12, y_pos, label, fontsize=10, color="#475569")
        ax.text(0.52, y_pos, val, fontsize=10, fontweight="bold", color="#0F172A")
        y_pos -= 0.032

    # 4. Second-Life Recommendations
    y_pos -= 0.015
    ax.text(0.08, y_pos, "2. Second-Life Suitability Assessment", fontsize=12, fontweight="bold", color="#1E293B")
    ax.plot([0.08, 0.92], [y_pos - 0.015, y_pos - 0.015], color="#CBD5E1", lw=1)

    y_pos -= 0.045
    ax.text(0.12, y_pos, "Suitable Applications:", fontsize=10.5, fontweight="bold", color="#16A34A")
    y_pos -= 0.026
    for item in recommendation.get("suitable", []):
        ax.text(0.15, y_pos, f"•  {item}", fontsize=9.5, color="#1E293B")
        y_pos -= 0.024

    y_pos -= 0.01
    ax.text(0.12, y_pos, "Unsuitable / Not Recommended:", fontsize=10.5, fontweight="bold", color="#DC2626")
    y_pos -= 0.026
    for item in recommendation.get("unsuitable", []):
        ax.text(0.15, y_pos, f"•  {item}", fontsize=9.5, color="#1E293B")
        y_pos -= 0.024

    y_pos -= 0.012
    why_text = recommendation.get("why", "")
    if why_text:
        ax.text(0.12, y_pos, f"Assessment Rationale: {why_text}", fontsize=9, color="#475569", style="italic")
        y_pos -= 0.035

    # 5. Disclaimer Box (Bottom)
    ax.add_patch(patches.FancyBboxPatch((0.08, 0.05), 0.84, 0.12, boxstyle="round,pad=0.015",
                                        facecolor="#FEF2F2", edgecolor="#FECACA", linewidth=1))
    ax.text(0.11, 0.145, "Important Safety & Technical Caveat:", fontsize=9, fontweight="bold", color="#991B1B")
    
    caveat_text = recommendation.get("caveat", "")
    full_caveat = (
        f"{caveat_text} "
        "Typical SOH error is ~3 points in laboratory testing on 4 NASA cells. "
        "The 80%, 70%, and 50% cutoffs are rules of thumb, not validated limits."
    )
    wrapped = textwrap.fill(full_caveat, width=95)
    ax.text(0.11, 0.075, wrapped, fontsize=7.5, color="#7F1D1D", va="bottom")

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
