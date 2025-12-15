import pandas as pd
import math

def main():
    # Hardcoded results from the execution logs
    results = [
        {"Model Size": "Tiny", "Data %": "10%", "Samples": 745, "Parameters": 5979, "Best Val Loss": 0.9601},
        {"Model Size": "Small", "Data %": "10%", "Samples": 745, "Parameters": 22187, "Best Val Loss": 0.9529},
        {"Model Size": "Medium", "Data %": "10%", "Samples": 745, "Parameters": 85323, "Best Val Loss": 0.9237},
        
        {"Model Size": "Tiny", "Data %": "20%", "Samples": 1491, "Parameters": 5979, "Best Val Loss": 0.9569},
        {"Model Size": "Small", "Data %": "20%", "Samples": 1491, "Parameters": 22187, "Best Val Loss": 0.9478},
        {"Model Size": "Medium", "Data %": "20%", "Samples": 1491, "Parameters": 85323, "Best Val Loss": 0.5248},
    ]

    df = pd.DataFrame(results)
    
    # Sort
    size_order = ["Tiny", "Small", "Medium"]
    df["Size_Rank"] = df["Model Size"].apply(lambda x: size_order.index(x))
    df = df.sort_values(by=["Size_Rank", "Samples"])
    df = df.drop(columns=["Size_Rank"])
    
    print("\n=== Scaling Laws POC Results ===\n")
    print(df.to_string(index=False))
    
    # Simple Analysis
    print("\n=== Analysis ===")
    
    # 1. Data Scaling (Fixed Model, Vary Data)
    print("\n[Data Scaling] Loss vs Data Size (Samples):")
    for size in size_order:
        subset = df[df["Model Size"] == size]
        if len(subset) > 1:
            l1 = subset.iloc[0]["Best Val Loss"]
            l2 = subset.iloc[1]["Best Val Loss"]
            d1 = subset.iloc[0]["Samples"]
            d2 = subset.iloc[1]["Samples"]
            
            # Power law: L = C * D^-alpha
            # log(L1/L2) = -alpha * log(D1/D2)
            # alpha = log(L1/L2) / log(D1/D2)  <-- Wait, loss decreases as data increases
            # L1 = C * D1^-a
            # L2 = C * D2^-a
            # L1/L2 = (D1/D2)^-a = (D2/D1)^a
            # log(L1/L2) = a * log(D2/D1)
            # a = log(L1/L2) / log(D2/D1)
            
            if l1 > 0 and l2 > 0 and d1 > 0 and d2 > 0:
                alpha = math.log(l1/l2) / math.log(d2/d1)
                print(f"  {size}: alpha = {alpha:.4f} (Loss {l1:.4f} -> {l2:.4f} as data doubled)")
            else:
                print(f"  {size}: Insufficient valid data")
    
    # 2. Model Scaling (Fixed Data, Vary Model)
    print("\n[Model Scaling] Loss vs Parameters:")
    for pct in ["10%", "20%"]:
        subset = df[df["Data %"] == pct].sort_values("Parameters")
        if len(subset) > 1:
            print(f"\n  Data {pct}:")
            prev_loss = None
            prev_params = None
            for _, row in subset.iterrows():
                curr_loss = row["Best Val Loss"]
                curr_params = row["Parameters"]
                if prev_loss is not None:
                    # beta = log(L_prev/L_curr) / log(P_curr/P_prev)
                    if curr_loss < prev_loss:
                        beta = math.log(prev_loss/curr_loss) / math.log(curr_params/prev_params)
                        print(f"    {prev_params} -> {curr_params} params: beta = {beta:.4f} (Loss {prev_loss:.4f} -> {curr_loss:.4f})")
                    else:
                        print(f"    {prev_params} -> {curr_params} params: Loss INCREASED ({prev_loss:.4f} -> {curr_loss:.4f}) - Inverse Scaling?")
                prev_loss = curr_loss
                prev_params = curr_params

if __name__ == "__main__":
    main()