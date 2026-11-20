def recommend(soh):
    caveat = ("SOH alone cannot show a battery is safe or compatible. Check voltage, "
              "capacity, power needs, chemistry, configuration and physical condition first.")
    if soh >= 80:
        return {"condition": "Healthy",
                "suitable": ["Original use (for example EV)", "Any lower-demand use"],
                "unsuitable": ["None ruled out by SOH alone"],
                "why": "Capacity is still high, so it can handle demanding use.",
                "caveat": caveat}
    if soh >= 70:
        return {"condition": "Aging",
                "suitable": ["Solar energy storage", "Home backup power", "Stationary energy storage"],
                "unsuitable": ["High-power use", "Fast charging", "EV driving"],
                "why": "It holds less energy, but slow and gentle cycling in a fixed place is easier on it.",
                "caveat": caveat}
    if soh >= 50:
        return {"condition": "Degraded",
                "suitable": ["Low-power, light-duty storage"],
                "unsuitable": ["EV use", "Critical backup power", "High-power loads"],
                "why": "Much of the capacity is gone, so only small, non-critical loads make sense.",
                "caveat": caveat}
    return {"condition": "Degraded",
            "suitable": ["Not recommended for reuse"],
            "unsuitable": ["All uses until professionally inspected"],
            "why": "Capacity is very low, and reuse is unlikely to be worthwhile or safe.",
            "caveat": caveat}

if __name__ == "__main__":
    for s in [90, 75, 65, 45]:
        r = recommend(s)
        print(s, "->", r["condition"], "| suitable:", r["suitable"], "| unsuitable:", r["unsuitable"])
