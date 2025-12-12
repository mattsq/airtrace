import math

def calculate_projection(current_val, current_loss, target_loss, exponent, metric_name):
    """
    Project required metric value using Power Law: L = C * X^(-exponent)
    X_target = X_current * (current_loss / target_loss)^(1/exponent)
    """
    if exponent <= 0:
        return float('inf')
    
    factor = (current_loss / target_loss) ** (1 / exponent)
    required_val = current_val * factor
    
    print(f"\nProjection for {metric_name} (Exponent alpha/beta = {exponent:.4f}):")
    print(f"  Current ({metric_name}): {current_val:,.0f} -> Loss: {current_loss:.4f}")
    print(f"  Target Loss: {target_loss:.4f}")
    print(f"  Required Factor: {factor:.2f}x")
    print(f"  Estimated Required {metric_name}: {required_val:,.0f}")
    return required_val

def main():
    target_loss = 0.1
    
    # Data Scaling (Based on Medium Model, 10% -> 20% Data)
    # This is the "optimistic" scaling law derived from the phase transition
    d1 = 745   # 10%
    l1 = 0.9237
    d2 = 1491  # 20%
    l2 = 0.5248
    
    alpha = -math.log(l2/l1) / math.log(d2/d1) # 0.8149
    
    print("=== Data Scaling Extrapolation ===")
    print("(Assuming Medium Model capacity is sufficient)")
    req_samples = calculate_projection(d2, l2, target_loss, alpha, "Samples")
    
    # Check against available data
    total_available = 7458 # 100%
    print(f"  Available Samples (100%): {total_available:,.0f}")
    if req_samples > total_available:
        print(f"  ⚠️ EXCEEDS AVAILABLE DATA by {(req_samples/total_available):.1f}x")
    else:
        print(f"  ✅ Within dataset limits ({(req_samples/total_available)*100:.1f}%)")

    # Model Scaling (Based on 20% Data, Small -> Medium Model)
    # This is the scaling law observed when data was sufficient
    p1 = 22187 # Small
    lp1 = 0.9478
    p2 = 85323 # Medium
    lp2 = 0.5248
    
    beta = -math.log(lp2/lp1) / math.log(p2/p1) # 0.4389
    
    print("\n=== Model Scaling Extrapolation ===")
    print("(Assuming Data is sufficient)")
    req_params = calculate_projection(p2, lp2, target_loss, beta, "Parameters")
    
    # Compare to standard sizes
    print(f"  Comparison: ~{req_params/1e6:.1f}M Parameters")

if __name__ == "__main__":
    main()
